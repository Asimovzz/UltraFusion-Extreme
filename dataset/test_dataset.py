import os, glob
import cv2
import torch
import numpy as np
import torch.utils.data as data
from PIL import Image, ImageOps
from torchvision.transforms import ToTensor


def get_color_and_struct(isrgb, input_img: torch.Tensor, ksize, sigmaX, c):  #input an RGB image

    input_img = input_img.squeeze().cpu().numpy().transpose(1, 2, 0)

    if isrgb==True:
        yuv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2YUV).astype(np.float32)
        y = np.expand_dims(yuv_img[:,:,0], axis=-1).astype(np.float64)
        u = np.expand_dims(yuv_img[:,:,1], axis=-1).astype(np.float32)
        v = np.expand_dims(yuv_img[:,:,2], axis=-1).astype(np.float32)
    else:
        y = input_img.astype(np.float64)
    #mu = gaussian_filter(y, ksize, ksize/6)
    mu = cv2.GaussianBlur(y, (ksize,ksize), sigmaX).astype(np.float64)
    mu_sq = mu * mu
    sigma = np.sqrt(np.absolute(cv2.GaussianBlur(y*y, (ksize,ksize), sigmaX) - mu_sq)).astype(np.float64)
    mu = np.expand_dims(mu, axis=-1)
    sigma = np.expand_dims(sigma, axis=-1)
    dividend = y.astype(np.float64) - mu
    divisor = sigma + c
    struct = dividend / divisor
    struct = struct.astype(np.float32)
    struct_norm = (struct - struct.min()) / (struct.max() - struct.min() + 1e-6)
    struct_norm = torch.from_numpy(struct_norm).permute(2, 0, 1)
    u = torch.from_numpy(u).permute(2, 0, 1)
    v = torch.from_numpy(v).permute(2, 0, 1)
    img_uv = torch.cat([u, v], dim=0)
    return struct_norm, img_uv


class TestDataset(data.Dataset):
    IMG_DIRS = {
        'UltraFusion': './data/UltraFusionBenchmark',
        'MEFB': './data/MEFB',
        'RealHDRV': './data/Real-HDRV-Deghosting-sRGB-Testing',
        'UltraFusionExtreme': './data/UltraFusion-Extreme-Cases',
        'UltraFusion-Extreme-Cases': './data/UltraFusion-Extreme-Cases',
    }

    def __init__(self, dataset, input_dir=None, max_long_edge=None):
        super(TestDataset, self).__init__()
        self.dataset = dataset
        self.img_dir = input_dir or self.IMG_DIRS.get(dataset)
        if self.img_dir is None:
            supported = ', '.join(sorted(self.IMG_DIRS.keys()))
            raise ValueError(f'Unknown dataset "{dataset}". Supported datasets: {supported}.')
        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f'Dataset directory not found: {self.img_dir}')

        self.ldr_list1 = []
        self.ldr_list2 = []
        self.file_name_list = []
        self.to_tensor = ToTensor()
        self.max_long_edge = max_long_edge

        self.scene_list = os.listdir(self.img_dir)
        self.scene_list.sort()
        for scene in self.scene_list:
            scene_dir = os.path.join(self.img_dir, scene)
            if os.path.isdir(scene_dir) and len(os.listdir(scene_dir)) > 0:
                ue_paths = glob.glob(os.path.join(scene_dir, '*ue.*'))
                oe_paths = glob.glob(os.path.join(scene_dir, '*oe.*'))
                if not ue_paths or not oe_paths:
                    continue
                self.ldr_list1.append(ue_paths[0])
                self.ldr_list2.append(oe_paths[0])
                self.file_name_list.append('{}_{}'.format(dataset, scene))
        

    def __getitem__(self, index):
        ldr1_path = self.ldr_list1[index]
        ldr2_path = self.ldr_list2[index]
        file_name = self.file_name_list[index]

        ldr1 = ImageOps.exif_transpose(Image.open(ldr1_path)).convert('RGB')
        ldr2 = ImageOps.exif_transpose(Image.open(ldr2_path)).convert('RGB')

        if ldr1.size != ldr2.size:
            if ldr1.size == (ldr2.size[1], ldr2.size[0]):
                ldr2 = ldr2.transpose(Image.Transpose.ROTATE_90)
            if ldr1.size != ldr2.size:
                ldr2 = ldr2.resize(ldr1.size, Image.Resampling.BICUBIC)

        W, H = ldr1.size

        if self.max_long_edge is not None and max(W, H) > self.max_long_edge:
            scale = self.max_long_edge / max(W, H)
            new_size = [max(1, int(W * scale)), max(1, int(H * scale))]
            ldr1 = ldr1.resize(new_size, Image.Resampling.BICUBIC)
            ldr2 = ldr2.resize(new_size, Image.Resampling.BICUBIC)
            W, H = ldr1.size

        if W * H >= 6000 * 4000:
            ldr1 = ldr1.resize([W // 4, H // 4])
            ldr2 = ldr2.resize([W // 4, H // 4])
        elif W * H >= 2000 *1500:
            ldr1 = ldr1.resize([W * 2 // 5, H * 2 // 5])
            ldr2 = ldr2.resize([W * 2 // 5, H * 2 // 5])

        ldr1 = self.to_tensor(ldr1)
        ldr2 = self.to_tensor(ldr2)

        return {
            'ue': ldr1,
            'oe': ldr2,
            'file_name': file_name
        }

    def __len__(self):
        return len(self.ldr_list1)
