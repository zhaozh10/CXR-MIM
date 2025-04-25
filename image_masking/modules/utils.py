
import torchvision.transforms.functional as F
import random
import torch
from PIL import Image
import numpy as np
import os
import datetime
import logging
import matplotlib.pyplot as plt

def RandomSeed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



def show_image(image, title=''):
    # image is [H, W, 3]
    assert image.shape[2] == 3
    plt.imshow(torch.clip(image * 255, 0, 255).int())
    plt.title(title, fontsize=16)
    plt.axis('off')
    return


class PairedTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, img1, img2):
        # 保存当前随机种子
        torch_state = torch.get_rng_state()
        random_state = random.getstate()

        # 对第一张图片应用变换
        img1 = self.transform(img1)

        # 恢复随机种子，确保相同的变换应用到第二张图片
        torch.set_rng_state(torch_state)
        random.setstate(random_state)
        img2 = self.transform(img2)

        return img1, img2
    


class ResizeWithAspectRatio:
    def __init__(self, size, interpolation=Image.BILINEAR):
        """
        Initialize the transform.

        Args:
        - size (tuple): Desired output size (height, width).
        - interpolation (int): Interpolation method. Default is PIL.Image.BILINEAR.
        """
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Apply the transform to an image.

        Args:
        - img (PIL Image or Tensor): Image to be resized.

        Returns:
        - PIL Image or Tensor: Resized image.
        """
        original_width, original_height = img.size
        target_height, target_width = self.size

        # Calculate new size to keep aspect ratio
        aspect_ratio = original_width / original_height

        if aspect_ratio > 1:  # Wider image
            new_width = target_width
            new_height = int(target_width / aspect_ratio)
        else:  # Taller image or square
            new_height = target_height
            new_width = int(target_height * aspect_ratio)

        # Resize the image
        resized_image = F.resize(img, (new_height, new_width), self.interpolation)

        # Calculate padding
        pad_left = (target_width - new_width) // 2
        pad_right = target_width - new_width - pad_left
        pad_top = (target_height - new_height) // 2
        pad_bottom = target_height - new_height - pad_top

        # Pad the resized image to the target size
        padded_image = F.pad(resized_image, (pad_left, pad_top, pad_right, pad_bottom), padding_mode='constant', fill=0)

        return padded_image

def setup_logging(work_dir,name):
    # Create a directory with the current timestamp
    current_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_dir = os.path.join(work_dir, f"{name}_{current_time}")
    os.makedirs(run_dir, exist_ok=True)

    # Set up logging to file and console
    log_file = os.path.join(run_dir, 'log.txt')
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ])

    return run_dir