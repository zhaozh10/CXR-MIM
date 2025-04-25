# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import random
import math
import torch
import numpy as np
import cv2




class RandomMasking:
    """ Masked Autoencoder with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, device='cpu'):
        super().__init__()

        # --------------------------------------------------------------------------
        self.patch_size=patch_size
        self.img_size=img_size
        self.num_patches=int((img_size/patch_size)**2)
        self.device=torch.device(device)
        # self.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_size
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, batch_size, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        # N, L, D = x.shape  # batch, length, dim
        N=batch_size
        L=self.num_patches
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=self.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        # The location of i-th (0-L) patch in ids_shuffle
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        # only keep first unmasked embeddings via indexing 
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

    def forward(self, x, mask_ratio):
        batch_size=x.shape[0]

        mask, ids_restore = self.random_masking(batch_size, mask_ratio)

        patch_size=self.patch_size
        # visualize the mask
        mask = mask.detach() #[batch_size, H*W]
        pix_mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 *3)  # [N, H*W, p*p*3]
        pix_mask = self.unpatchify(pix_mask)  # 1 is removing, 0 is keeping
        pix_mask = torch.einsum('nchw->nhwc', pix_mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', x)

        # masked image
        im_masked = x * (1 - pix_mask)
 
        return im_masked, mask, ids_restore



class GazeMasking:
    """ Gaze conditioned Masking
    """
    def __init__(self, img_size=224, patch_size=16, device='cpu'):
        super().__init__()

        # --------------------------------------------------------------------------
        self.patch_size=patch_size
        self.img_size=img_size
        self.num_patches=int((img_size/patch_size)**2)
        self.device=torch.device(device)


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

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_size
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def binary_seg(self,x_attn, mode='otsu'):
        batchsize=x_attn.shape[0]

        attention_gray = (x_attn.mean(dim=1)*255).numpy().astype(np.uint8)

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
        indices = torch.arange(L).unsqueeze(0).expand(N, -1)

        # 使用广播和比较操作生成布尔矩阵
        bool_matrix = indices < len_keep.unsqueeze(dim=-1)

        ids_keep=torch.mul(ids_shuffle,bool_matrix)
        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]


        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=self.device)
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
        noise = torch.rand(N, L, device=self.device)  # noise in [0, 1]

        clinical_noise=torch.mul(noise,clinical_significant_mask)
        non_clinical_noise=torch.mul(noise,~clinical_significant_mask)

        len_clinical=clinical_significant_mask.sum(dim=-1)

        len_non=L-len_clinical

        # len_keep_clinical = (len_clinical * (1 - clinical_ratio)).type(torch.int8)
        # len_keep_non=(L*(1-mask_ratio)-len_keep_clinical).type(torch.int8)
        len_mask_clinical = (len_clinical *  clinical_ratio).type(torch.uint8)
        len_mask_non=(L*mask_ratio-len_mask_clinical).type(torch.uint8)

        clinical_mask, ids_keep_clinical=self.noise2mask(clinical_noise,len_mask_clinical)
        non_clinical_mask, ids_keep_non=self.noise2mask(non_clinical_noise,len_mask_non)

        # # 使用 torch.nonzero 获取非零元素的索引
        # ids_restore=np.zeros_like(ids_keep_clinical)

        # # 从矩阵中提取所有非零值
        # for i in range(N):
        #     ids_restore_clinical_indices = torch.nonzero(ids_keep_clinical[i,:])
        #     ids_restore_non_indices = torch.nonzero(ids_keep_non[i,:])
        #     nonzero_clinical = ids_keep_clinical[i, ids_restore_clinical_indices]
        #     nonzero_non = ids_keep_non[i,ids_restore_non_indices]
        #     nonzero=torch.cat((nonzero_clinical,nonzero_non),dim=0).squeeze()
        #     ids_restore[i,:len(nonzero)]=nonzero

        # 进行并集操作
        union_mask = clinical_mask | non_clinical_mask

        zero_positions = (union_mask == 0).nonzero(as_tuple=False)  # 结果形状是 [N, 3]
        one_positions = (union_mask == 1).nonzero(as_tuple=False) 
        ids_visible=zero_positions[:,1].view(N, -1)
        ids_invisible=one_positions[:,1].view(N, -1)
        return union_mask,ids_visible,ids_invisible




    def forward(self, x, mask_ratio=0.8, clinical_ratio=0.5, preload=False):
        assert isinstance(x,dict), print(type(x))
        
        if preload:
            x_img=x['img']
            x_foreground=x['foreground']
        else:
            x_img=x['img']
            x_attn=x['attention']
            x_foreground=self.binary_seg(x_attn)
        
        
        
        # batch_size=x_img.shape[0]
        # N, L, D = x.shape  #
        # mask, ids_restore = self.create_mask(x_foreground.clone(), mask_ratio,clinical_ratio)
        mask, ids_visible,ids_invisible = self.create_mask(x_foreground.clone(), mask_ratio, clinical_ratio)
        # ids_restore = torch.argsort(ids_visible, dim=1)
        # dummy=torch.randn(2,196,768)
        # torch.gather(dummy, dim=1, index=ids_visible.unsqueeze(-1).repeat(1,1,768)).shape
        # x_masked = torch.gather(dummy, dim=1, index=ids_visible)

        ids_concat=torch.concat([ids_visible,ids_invisible],dim=1)
        ids_restore=torch.argsort(ids_concat,dim=1)


        patch_size=self.patch_size
        # visualize the mask
        mask = mask.detach() #[batch_size, H*W]
        pix_mask = mask.unsqueeze(-1).repeat(1, 1, patch_size**2 *3)  # [N, H*W, p*p*3]
        pix_mask = self.unpatchify(pix_mask)  # 1 is removing, 0 is keeping
        pix_mask = torch.einsum('nchw->nhwc', pix_mask).detach().cpu()

        x = torch.einsum('nchw->nhwc', x_img)

        # masked image
        im_masked = x * (1 - pix_mask)
 
        return im_masked, mask, x_foreground, ids_restore


class BlockMasking:
    def __init__(
            self, img_size, patch_size=16, device='cpu'):
        self.device=torch.device(device)
        self.patch_size=patch_size
        self.height = int(img_size/patch_size)
        self.width = int(img_size/patch_size)
        self.num_patches = self.height * self.width
        self.num_masking_patches=None
        self.min_num_patches=None
        self.max_num_patches=None
        # max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = None
        

    def __repr__(self):
        repr_str = "Generator(%d, %d -> [%d ~ %d], max = %d, %.3f ~ %.3f)" % (
            self.height, self.width, self.min_num_patches, self.max_num_patches,
            self.num_masking_patches, self.log_aspect_ratio[0], self.log_aspect_ratio[1])
        return repr_str

    # def get_shape(self):
    #     return self.height, self.width
    
    def unpatchify(self, x):
        """
        x: (batch_size, H,W,768)
        """

        p = self.patch_size
        h=w= x.shape[1]
        # h = w = int(x.shape[1]**.5)
        # assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for attempt in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # Overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    for i in range(top, top + h):
                        for j in range(left, left + w):
                            if mask[i, j] == 0:
                                mask[i, j] = 1
                                delta += 1

                if delta > 0:
                    break
        return delta

    def __call__(self,x, mask_ratio=0.4, min_num_patches=4, max_num_patches=None,
            min_aspect=0.3, max_aspect=None):
        
        batch_size=x.shape[0]
        self.num_masking_patches = self.num_patches*mask_ratio
        self.min_num_patches = min_num_patches
        self.max_num_patches = self.num_patches*mask_ratio if max_num_patches is None else max_num_patches

        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

        mask = np.zeros(shape=(batch_size,self.height,self.width), dtype=np.int64)
        for i in range(batch_size):
            mask_count = 0
            while mask_count < self.num_masking_patches:
                max_mask_patches = self.num_masking_patches - mask_count
                max_mask_patches = min(max_mask_patches, self.max_num_patches)

                delta = self._mask(mask[i], max_mask_patches)
                if delta == 0:
                    break
                else:
                    mask_count += delta
        mask=torch.from_numpy(mask)



        pix_mask = mask.unsqueeze(-1).repeat(1, 1,1, self.patch_size**2 *3)# (1, H, W, p*p*3)
        pix_mask = self.unpatchify(pix_mask)  # 1 is removing, 0 is keeping
        pix_mask = torch.einsum('nchw->nhwc', pix_mask).detach().cpu()


        x = torch.einsum('nchw->nhwc', x)

        im_masked = x * (1 - pix_mask)
        return im_masked,mask