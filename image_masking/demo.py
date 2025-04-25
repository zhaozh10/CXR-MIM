from modules.utils import ResizeWithAspectRatio,show_image
from modules.mask import BlockMasking,RandomMasking,GazeMasking
from torchvision import transforms as T
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import cv2

data_prefix="../data/reflacx-1.0.0/"
gaze_demo=["P106R223835.jpg","P202R811865.jpg","P300R460237.jpg"]

vis_trans={"PIL":
           T.Compose([
            ResizeWithAspectRatio(size=(224, 224)),
                ]),
            "Tensor":
            T.Compose([
            # T.RandomHorizontalFlip(),
            ResizeWithAspectRatio(size=(224, 224)),
            T.ToTensor(),
                ])
           }
mask_ratio=0.75
clinical_ratio=0.5
choice=0

img_path="xrays/P106R223835.jpg"
attention_path="attentions/P106R223835.png"
img=Image.open(img_path).convert('RGB')
attention=Image.open(attention_path).convert('RGB')
resized_img=vis_trans["Tensor"](img).unsqueeze(0)
resized_attention=vis_trans["Tensor"](attention).unsqueeze(0)

random_masking=RandomMasking(img_size=224,patch_size=16,device='cpu')
gaze_masking=GazeMasking(img_size=224,patch_size=16,device='cpu')
im_gaze_masked,mask_gaze, foreground, ids_restore_gaze=gaze_masking.forward({'img':resized_img.repeat(2,1,1,1),'attention':resized_attention.repeat(2,1,1,1)},mask_ratio=0.8,clinical_ratio=0.5,preload=False)

im_random_masked, mask_random,ids_restore=random_masking.forward(resized_img,mask_ratio=0.75)
im_gaze_masked,mask_gaze, foreground, ids_restore_gaze=gaze_masking.forward({'img':resized_img,'attention':resized_attention},mask_ratio=mask_ratio,clinical_ratio=clinical_ratio,preload=False)

binary_otsu=np.array(foreground)

# 将二值化结果转换为PIL图像
binary_otsu_img = Image.fromarray(binary_otsu.squeeze())

# 创建半透明模板
binary_otsu_img = binary_otsu_img.convert("RGBA")

# 将二值化结果转换为RGBA通道
otsu_alpha = binary_otsu_img.split()[0].point(lambda p: p > 0 and 128)  # 设置透明度

# 将分割结果叠加到目标图像上
otsu_overlay = Image.composite(vis_trans["PIL"](img).convert("RGBA"), binary_otsu_img, otsu_alpha)


plt.rcParams['figure.figsize'] = [16, 16]
plt.subplot(1, 4, 1)
show_image(torch.einsum('nchw->nhwc', resized_img).detach().cpu()[0], "original")

plt.subplot(1, 4, 2)
plt.imshow(otsu_overlay)
plt.title('Otsu Segmentation Overlay')
plt.axis('off')

plt.subplot(1, 4, 3)
show_image(im_random_masked[0], f"random \nmasking ratio {mask_ratio}")

plt.subplot(1, 4, 4)
show_image(im_gaze_masked[0], f"masking ratio {mask_ratio} \nclinical ratio {clinical_ratio}")

plt.tight_layout()
plt.show()

# # make the plt figure larger
# plt.rcParams['figure.figsize'] = [12, 24]

# plt.subplot(1, 2, 1)
# show_image(torch.einsum('nchw->nhwc', resized_img).detach().cpu()[0], "original")

# plt.subplot(1, 2, 2)
# show_image(im_gaze_masked[0], f"Mask ratio {mask_ratio} clinical ratio {clinical_ratio}")

# plt.show()
