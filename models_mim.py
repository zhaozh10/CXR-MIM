# 2022.3.3-Changed for building LocalMIM
#          Huawei Technologies Co., Ltd. <foss@huawei.com>
#
# Modified from MAE by Haoqing Wang
# MAE: https://github.com/facebookresearch/mae
#
# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the CC-by-NC license found in the
# LICENSE file in the root directory of this source tree.

from functools import partial
import math
import cv2
import torch
import torch.nn as nn
from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed
import numpy as np
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    A LayerNorm variant, popularized by Transformers, that performs point-wise mean and
    variance normalization over the channel dimension for inputs that have shape
    (batch_size, channels, height, width).
    https://github.com/facebookresearch/ConvNeXt/blob/d1fa8f6fef0a165b27399986cc2bdacc92777e40/models/convnext.py#L119  # noqa B950
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


class MAE_Decoder(nn.Module):
    def __init__(self, inp_dim, embed_dim=256, out_dim=27, scale=1., num_patches=196, depth=1, num_heads=8, mlp_ratio=4., qkv_bias=False, norm_layer=nn.LayerNorm):
        super().__init__()
        self.num_patches = num_patches
        self.embed = nn.Linear(inp_dim, embed_dim, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+self.num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=qkv_bias, norm_layer=norm_layer) for _ in range(depth)])
        self.norm = norm_layer(embed_dim)

        # pred head
        hidden = embed_dim
        if scale == 4.0:
            layers = [nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2),
                      LayerNorm(embed_dim//2),
                      nn.GELU(),
                      nn.ConvTranspose2d(embed_dim//2, embed_dim//4, kernel_size=2, stride=2)]
            hidden = embed_dim//4
        elif scale == 2.0:
            layers = [nn.ConvTranspose2d(embed_dim, embed_dim//2, kernel_size=2, stride=2)]
            hidden = embed_dim//2
        elif scale == 1.0:
            layers = []
        elif scale == 0.5:
            layers = [nn.AvgPool2d(kernel_size=2, stride=2)]
        else:
            raise NotImplementedError(f"scale_factor={scale} is not supported yet.")
        layers.append(nn.Conv2d(hidden, out_dim, kernel_size=1))
        self.pred = nn.Sequential(*layers)

        self.initialize_weights()

    def initialize_weights(self):
        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize position embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, ids_restore):
        # embed tokens
        x = self.embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1]+1-x.shape[1], 1)
        # without consideration of cls token before unshuffle
        x_ = torch.cat([x[:, 1:], mask_tokens], dim=1)
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # append cls token
        x = torch.cat([x[:, :1], x_], dim=1)  

        # add pos embed
        x = x + self.pos_embed

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)  # [B, L, d]

        # predictor projection
        H = W = int(self.num_patches**0.5)
        # [batch, num_patch-1, hidden_dim] --> [batch, hidden_dim, num_patch]
        x = x[:, 1:].transpose(1, 2).reshape(x.size(0), -1, H, W)
        x = self.pred(x)
        x = x.flatten(2, 3).transpose(1, 2)

        return x


class HOGLayer(nn.Module):
    def __init__(self, nbins, pool, bias=False, max_angle=math.pi, stride=1, padding=1, dilation=1):
        super(HOGLayer, self).__init__()
        self.nbins = nbins

        self.conv = nn.Conv2d(1, 2, 3, stride=stride, padding=padding, dilation=dilation, padding_mode='reflect', bias=bias)
        mat = torch.FloatTensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
        mat = torch.cat((mat[None], mat.t()[None]), dim=0)
        self.conv.weight.data = mat[:, None, :, :]

        self.max_angle = max_angle
        self.pooler = nn.AvgPool2d(pool, stride=pool, padding=0, ceil_mode=False, count_include_pad=True)

    @ torch.no_grad()
    def forward(self, x):  # [B, 1, 224, 224]
        gxy = self.conv(x)

        # 2. Mag/ Phase
        mag = gxy.norm(dim=1)
        norm = mag[:, None, :, :]
        # phase = torch.atan2(gxy[:, 0, :, :], gxy[:, 1, :, :])
        phase = torch.atan2(gxy[:, 1, :, :], gxy[:, 0, :, :])

        # 3. Binning Mag with linear interpolation
        phase_int = phase/self.max_angle*self.nbins
        phase_int = phase_int[:, None, :, :]

        n, c, h, w = gxy.shape
        out = torch.zeros((n, self.nbins, h, w), dtype=torch.float, device=gxy.device)
        out.scatter_(1, phase_int.floor().long() % self.nbins, norm)

        hog = self.pooler(out)
        hog = nn.functional.normalize(hog, p=2, dim=1)
        return hog


class CXRMIM(nn.Module):

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=1024, depth=1, num_heads=8, decoder_embed_dim=512,
                 decoder_depth=1, decoder_num_heads=16, mlp_ratio=4., norm_layer=nn.LayerNorm, hog_nbins=9, hog_bias=False, gaze_weighted=False, \
                      preload=False, ID_list=[1, 3, 9, 11], scale_list=[4.0, 2.0, 1.0, 0.5], **kwargs):
        super().__init__()
        # MIM encoder specifics
        self.preload=preload
        self.patch_size=patch_size
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches=num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1+num_patches, embed_dim), requires_grad=False)  
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer) for _ in range(depth)])
        self.ID = ID_list
        print(f"Source layers-1: {self.ID}")
        self.scale = scale_list
        print(f"Scale {self.scale}")
        self.norm = nn.ModuleList([norm_layer(embed_dim) for _ in range(len(self.ID))])
        self.gaze_weighted=gaze_weighted
        print(f"Gaze weighted loss {self.gaze_weighted}")
        if self.gaze_weighted:
            self.gaze_weight_temperature=torch.tensor(0.1)
        self.initialize_weights()
        print(f"decoder depth {decoder_depth} decoder_num_heads {decoder_num_heads}")
        self.decoder = nn.ModuleList([
            MAE_Decoder(embed_dim, decoder_embed_dim, in_chans*hog_nbins, s, num_patches, decoder_depth, decoder_num_heads, mlp_ratio, True, norm_layer)
            for s in self.scale])

        # target
        self.hog_pool_list=[int(self.patch_size/k) for k in self.scale]
        self.hog_enc = nn.ModuleList([HOGLayer(nbins=hog_nbins, pool=k, bias=hog_bias) for k in self.hog_pool_list])
        for hog_enc in self.hog_enc:
            for param in hog_enc.parameters():
                param.requires_grad = False

    def initialize_weights(self):
        # initialization
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches**.5), cls_token=True)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        torch.nn.init.normal_(self.cls_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def hog_feat(self, imgs, k):  # [B, 3, 224, 224]
        """
        imgs: (N, 3, H, W)
        x: (N, L, d)
        """
        hog_R = self.hog_enc[k](imgs[:, :1, :, :])  # [B, nbins, h, w]
        hog_G = self.hog_enc[k](imgs[:, 1:2, :, :])  # [B, nbins, h, w]
        hog_B = self.hog_enc[k](imgs[:, 2:, :, :])  # [B, nbins, h, w]
        hog_feat = torch.cat([hog_R, hog_G, hog_B], 1)  # [B, 3*nbins, h, w]
        hog_feat = hog_feat.flatten(2, 3).transpose(1, 2)
        return hog_feat

   
    def gaze_patchify(self, imgs):
        """
        imgs: (N, H, W)
        x: (N, L, patch_size**2)
        """
        p = self.patch_size
        assert imgs.shape[1] == imgs.shape[2] and imgs.shape[1] % p == 0

        h = w = imgs.shape[1] // p
        x = imgs.reshape(shape=(imgs.shape[0], h, p, w, p))
        x = torch.einsum('nhpwq->nhwpq', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        x=(x/255).type(torch.uint8)
        return x
    
    def binary_seg(self,x_attn, mode='otsu'):
        batchsize=x_attn.shape[0]
        # This line only for [0,1] float type [3,224,224] attention map
        attention_gray = (x_attn.cpu().mean(dim=1)*255).numpy().astype(np.uint8)

        results = []
        if mode=='otsu':
            for i in range(batchsize):
                threshold, binary_foreground = cv2.threshold(attention_gray[i], 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                results.append(binary_foreground)
        else:
            pass
        binary_foreground=np.array(results)
        return  torch.from_numpy(binary_foreground)
    
    def noise2mask(self,noise,len_keep):
        N,L=noise.shape
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1,descending=True)  # descend: large is keep, small is remove
        # The location of i-th (0-L) patch in ids_shuffle
        ids_restore = torch.argsort(ids_shuffle, dim=1)


        # 生成索引矩阵，形状为[N, L]
        indices = torch.arange(L,device=noise.device).unsqueeze(0).expand(N, -1)

        # 使用广播和比较操作生成布尔矩阵
        bool_matrix = indices < len_keep.unsqueeze(dim=-1)

        ids_keep=torch.mul(ids_shuffle,bool_matrix)
        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]


        # generate the binary mask: 0 is keep, 1 is remove
        # mask = torch.ones([N, L], device=self.device)
        mask = torch.ones([N, L],device=noise.device)
        mask=torch.mul(mask,bool_matrix)
        # unshuffle 1s to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore).type(torch.uint8)
        return mask,ids_keep

    def create_mask(self, x_foreground, mask_ratio, clinical_ratio):

        N=x_foreground.shape[0]
        L=self.num_patches

        # consider it a clinical patch only when gaze map covers more than half of its area
        significant_threshold=(self.patch_size * self.patch_size) // 2

        x_foreground=self.gaze_patchify(x_foreground)
        x_foreground=x_foreground.sum(dim=-1)
        clinical_significant_mask=x_foreground>=significant_threshold


        # generate random noise following uniform distribution
        noise = torch.rand(N, L,device=x_foreground.device)  # noise in [0, 1]

        clinical_noise=torch.mul(noise,clinical_significant_mask)
        non_clinical_noise=torch.mul(noise,~clinical_significant_mask)

        len_clinical=clinical_significant_mask.sum(dim=-1)

        len_non=L-len_clinical

        len_mask_clinical = (len_clinical *  clinical_ratio).type(torch.uint8)
        len_mask_non=(int(L*mask_ratio)-len_mask_clinical).type(torch.uint8)

        post_len_mask_non=torch.where(len_mask_non>len_non,len_non,len_mask_non)
        post_len_mask_clinical=(int(L*mask_ratio)-post_len_mask_non).type(torch.uint8)

        clinical_mask, ids_keep_clinical=self.noise2mask(clinical_noise,post_len_mask_clinical)
        non_clinical_mask, ids_keep_non=self.noise2mask(non_clinical_noise,post_len_mask_non)


        # 进行并集操作
        union_mask = clinical_mask | non_clinical_mask

        zero_positions = (union_mask == 0).nonzero(as_tuple=False)  # 结果形状是 [N, 2]
        one_positions = (union_mask == 1).nonzero(as_tuple=False) 
        ids_visible=zero_positions[:,1].view(N, -1)
        ids_invisible=one_positions[:,1].view(N, -1)

        return union_mask,ids_visible,ids_invisible

    def gaze_masking(self,x: torch.Tensor, x_foreground:torch.Tensor, mask_ratio: float = 0.75, clinical_ratio=0.5):
    
        N, L, D = x.shape  # batch, length, dim
        mask, ids_visible, ids_invisible = self.create_mask(x_foreground.clone(), mask_ratio,clinical_ratio)
        mask=mask.to(x.device)
        ids_visible=ids_visible.to(x.device)
        ids_invisible=ids_invisible.to(x.device)

        ids_concat=torch.concat([ids_visible,ids_invisible],dim=1)
        ids_restore=torch.argsort(ids_concat,dim=1)

        x_masked = torch.gather(
            x, dim=1, index=ids_visible.unsqueeze(-1).repeat(1,1,D))

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, x_foreground, mask_ratio, clinical_ratio):
        
        # embed patches
        x = self.patch_embed(x)  # [B, num_patches, d]
        # add pos embed
        x = x + self.pos_embed[:, 1:]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.gaze_masking(x,x_foreground, mask_ratio, clinical_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        latent = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            if i in self.ID:
                latent.append(self.norm[self.ID.index(i)](x))

        return latent, mask.float(), ids_restore


    def recal_mask(self, mask, k):
        B, L, s = mask.size(0), mask.size(1), self.scale[k]
        H = W = int(L**.5)
        if s >= 1.:
            s = int(s)
            mask = mask.reshape(B, H, W).unsqueeze(3).unsqueeze(2).repeat(1, 1, s, 1, s).reshape(B, -1)
        else:
            s = int(1/s)
            mask = mask.reshape(B, H//s, s, H//s, s).transpose(2, 3).mean((-2, -1)).reshape(B, -1)

        return mask
    
    def recal_mask_attn(self,mask,attn,k):
        B, L, s = mask.size(0), mask.size(1), self.scale[k]
        pooled_attn=F.avg_pool2d(attn,kernel_size=int(self.patch_size/s),padding=0)
        H = W = int(L**.5)
        if s >= 1.:
            s = int(s)
            mask = mask.reshape(B, H, W).unsqueeze(3).unsqueeze(2).repeat(1, 1, s, 1, s).reshape(B, -1)
            
        else:
            s = int(1/s)
            mask = mask.reshape(B, H//s, s, H//s, s).transpose(2, 3).mean((-2, -1)).reshape(B, -1)

        return mask,pooled_attn

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = [self.hog_feat(imgs, i) for i in range(len(self.hog_enc))]
        # for k=[16/4, 16/2, 16/1, 16/0.5] (patch_size/scale)
        # target = [batchsize, (224/pach_size*scale)**2, 3*nbins]
        loss = 0.
        for k in range(len(pred)):
            M = self.recal_mask(mask, k)
            loss += (((pred[k]-target[k])**2).mean(dim=-1)*M).sum()/M.sum()

        return loss
    
    def weighted_forward_loss(self, imgs, pred, mask,attention):
        """
        imgs: [N, 3, H, W]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        N,C,H,W=imgs.shape
        target = [self.hog_feat(imgs, i) for i in range(len(self.hog_enc))]
        # for k=[16/4, 16/2, 16/1, 16/0.5] (patch_size/scale)
        # target = [batchsize, (224/pach_size*scale)**2, 3*nbins]
        loss = 0.
        for k in range(len(pred)):
            M,pooled_attn = self.recal_mask_attn(mask, attention,k)
            pooled_attn=(pooled_attn.flatten(1,2))*M

            attn_weights=torch.softmax(pooled_attn/self.gaze_weight_temperature,dim=-1)

            attn_weights=attn_weights+M

            hog_loss=((pred[k]-target[k])**2).mean(dim=-1)
            loss += (hog_loss * attn_weights).sum() / attn_weights.sum()

        return loss


    
    def forward(self, x, mask_ratio=0.75, clinical_ratio=0.5):  # [B, C, H, W]

        imgs=x['img']
        imgs_attn=x['attention']
        if self.preload:
            imgs_foreground=x['foreground']
        else:
            print("online extraction!")
            imgs_foreground=self.binary_seg(imgs_attn)

        # latent: A list with the same length of self.ID, with the shape of each element = [batch_size, num_patch*(1-mask_ratio)+1, embed_dim]
        latent, mask, ids_restore = self.forward_encoder(imgs, imgs_foreground, mask_ratio, clinical_ratio)
        pred = [self.decoder[i](latent[i], ids_restore) for i in range(len(latent))]
        if self.gaze_weighted:
            loss=self.weighted_forward_loss(imgs, pred, mask, imgs_attn)
        else:
            loss = self.forward_loss(imgs, pred, mask)
        return loss



def extract_vit_backbone(ckpt, source: str='mae',prefix=None)->callable:
    # prefix='encoder.layers
    state_dict=ckpt
    if prefix !=None:
         for k in list(state_dict.keys()):
            if k.startswith(f'{prefix}.'):
                # print(k)
                if not k.startswith(f'{prefix}.fc'):
                    # remove prefix
                    state_dict[k[len(f"{prefix}."):]] = state_dict[k]
            # del掉不是backbone的部分
            del state_dict[k]


    if source == None:
        for k in list(state_dict.keys()):
            if k.startswith('head'):
                del state_dict[k]
        return state_dict
    elif source=='mae':
        for k in list(state_dict.keys()):
            if k.startswith('patch_embed'):
                state_dict[k.replace('projection','proj')]=state_dict[k]
                del state_dict[k]
            elif k.startswith('layers'):
                layer_num=eval(k.split('.')[1])
                new_key='blocks'+k[len("layers"):]
                new_key=new_key.replace('.ln','.norm').replace('.ffn.layers.0.0.','.mlp.fc1.').replace('.ffn.layers.1','.mlp.fc2')
                state_dict[new_key]=state_dict[k]
                del state_dict[k]
            elif k.startswith('ln1'):
                state_dict[k.replace('ln1','norm')]=state_dict[k]
                del state_dict[k]
        return state_dict




def cxrmim_vit_base_patch16(**kwargs):
    model = CXRMIM(
        patch_size=16,
        embed_dim=768,
        depth=12,
        num_heads=12,
        decoder_embed_dim=256,
        mlp_ratio=4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    ckpt=torch.load('./preTrain/vit_base_p16_224_timm.pth')
    ckpt_dict=extract_vit_backbone(ckpt,source=None)
    model.load_state_dict(ckpt_dict,strict=False)
    
    return model




