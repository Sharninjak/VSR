import glob
import itertools
import pathlib
import random
from typing import List

import torch.utils.data as data
import numpy as np
import torchvision.transforms
from PIL import Image, ImageFilter


class ImageSequenceDataset(data.Dataset):
    want_shuffle = True # 是否打乱
    pix_type = 'rgb' # 图像类型

    def __init__(self, index_file, patch_size, scale_factor, augment, seed=0):
        self.dataset_base = pathlib.Path(index_file).parent # 获取路径的父目录
        self.sequences = [i for i in open(index_file, 'r', encoding='utf-8').read().split('\n')
                          if i if not i.startswith('#')]
        self.patch_size = patch_size
        self.scale_factor = scale_factor
        self.augment = augment
        self.rand = random.Random(seed) # 随机数生成器
        self.transform = torchvision.transforms.ToTensor() # 将PIL Image或NumPy数组转换为PyTorch张量

    def _load_sequence(self, path):
        # 查找指定目录下的所有.png
        path = self.dataset_base / "sequences" / path
        files = glob.glob("*.png", root_dir=path)
        assert len(files) > 1
        images = [Image.open(file) for file in files]
        # 检查所有图片的尺寸是否相同
        if not all(i.size != images[0].size for i in images[1:]):
            raise ValueError("sequence has different dimensions")
        return images

    def _prepare_images(self, images: List[Image.Image]):
        # 随机裁剪图像
        w, h = images[0].size # 以第一张图像的尺寸作为基准宽度(w)和高度(h)
        f = self.scale_factor
        sw, sh = self.patch_size # 基础裁剪尺寸
        sw, sh = sw * f, sh * f 
        # 所有图像的高度和宽度都大于等于要裁剪的区域大小
        assert h >= sh and w >= sw
        # 随机确定裁剪起始位置
        dh, dw = self.rand.randint(0, h - sh), self.rand.randint(0, w - sw)
        # 裁剪的四个角的坐标
        images_s = [i.crop((dw, dh, dw + sw, dh + sh)) for i in images]
        return images_s

    # 不同类型的图像变换操作组合
    trans_groups = {
        'none': [None],
        'rotate': [None, Image.ROTATE_90, Image.ROTATE_180, Image.ROTATE_270],
        'mirror': [None, Image.FLIP_LEFT_RIGHT],
        'flip': [None, Image.FLIP_LEFT_RIGHT, Image.FLIP_TOP_BOTTOM, Image.ROTATE_180],
        'all': [None] + [e.value for e in Image.Transpose] # 所有转换操作的名称
    }

    trans_names = [e.name for e in Image.Transpose]

    def _augment_images(self, images: List[Image.Image], trans_mode='all'):
        # 数据增强
        trans_action = 'none'
        trans_op = self.rand.choice(self.trans_groups[trans_mode])
        if trans_op is not None:
            images = [i.transpose(trans_op) for i in images]
            trans_action = self.trans_names[trans_op]
        return images, trans_action

    # 缩放图像的滤波器 BILINEAR:双线性插值 BICUBIC:双三次插值 LANCZOS:Lanczos插值
    scale_filters = [Image.BILINEAR, Image.BICUBIC, Image.LANCZOS]

    # 随机选择滤波器缩放图像
    def _scale_images(self, images: List[Image.Image]):
        f = self.scale_factor
        return [i.resize((i.width // f, i.height // f), self.rand.choice(self.scale_filters)) for i in images]

    # 随机选择降质操作
    def _degrade_images(self, images: List[Image.Image]):

        degrade_action = None
        decision = self.rand.randrange(4) # 生成0-3之间的随机整数
        if decision == 1:
            degrade_action = 'box' # BoxBlur滤波器
            percent = 0.5 + 0.5 * self.rand.random() # 随机0.5-1之间的浮点数
            # blend将原图与模糊图像按比例混合 filter()应用模糊滤镜 
            images = [Image.blend(j, j.copy().filter(ImageFilter.BoxBlur(1)), percent) for j in images]
        elif decision == 2:
            degrade_action = 'gaussian' # 高斯模糊
            radius = self.rand.random() # 0-1间浮点数，高斯模糊的半径
            images = [j.filter(ImageFilter.GaussianBlur(radius)) for j in images]
        elif decision == 3:
            degrade_action = 'halo' # Halo滤波器
            percent = 0.5 + 0.5 * self.rand.random() # 随机0.5-1之间的浮点数
            # 缩小再放大图像制造锐度损失和光晕效果
            images = [Image.blend(i,
                                  i.resize((i.width // 2, i.height // 2), resample=Image.LANCZOS)
                                  .resize(i.size, resample=Image.BILINEAR), percent)
                      for i in images]

        return images, degrade_action

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 当需要获取一个样本时被调用
        sequence = self._load_sequence(self.sequences[idx])
        sequence = self._prepare_images(sequence)  # crop to requested size
        original, _ = self._augment_images(sequence)  # flip and rotates
        lfs_pred = [np.array(lf.resize((lf.width // self.scale_factor, lf.height // self.scale_factor), Image.LANCZOS))
                    for lf in original[1::2]]
        lfs_deg = self._scale_images(original[::2])
        # lfs_deg, _ = self._degrade_images(lfs_deg)
        degraded = [i for i in itertools.zip_longest(lfs_deg, lfs_pred) if i is not None]
        original = [self.transform(i) for i in original]
        degraded = [self.transform(i) for i in degraded]
        return original, degraded
