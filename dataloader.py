import os
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import random
import cv2
from torchvision import transforms


class AugmentedDentalDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): 根目录路径（直接包含 model001, model002, ... 的目录）
            transform (callable, optional): 可选的变换操作（本类内部已定义 ToTensor）
        """
        self.root_dir = root_dir
        # 注意：这里我们不使用外部传入的 transform，而是固定使用 ToTensor（你也可以改成支持外部传入）
        self.transform = transforms.Compose([
            transforms.ToTensor(),  # 自动将 [0,255] uint8 转为 [0,1] float
        ])
        self.samples = []
        self._discover_samples()
        print(f"成功加载 {len(self.samples)} 个样本")

    def _discover_samples(self):
        # 直接在 root_dir 下查找所有 modelXXX 文件夹（例如 model001, model123 等）
        model_dirs = glob.glob(os.path.join(self.root_dir, "model*"))
        
        # 按数字排序（提取 model 后的数字）
        def extract_number(path):
            basename = os.path.basename(path)
            # 去掉 "model" 前缀，剩下的应为纯数字（如 "001", "123"）
            num_str = basename.replace("model", "")
            try:
                return int(num_str)
            except ValueError:
                return float('inf')  # 无法解析的放最后
        
        model_dirs = sorted(model_dirs, key=extract_number)

        for model_dir in model_dirs:
            csv_path = os.path.join(model_dir, "3.csv")
            if not os.path.exists(csv_path):
                print(f"跳过缺失 3.csv 的目录: {model_dir}")
                continue

            # 获取所有 .bmp 文件
            bmp_files = glob.glob(os.path.join(model_dir, "*.bmp"))
            if not bmp_files:
                print(f"目录 {model_dir} 中没有 BMP 文件")
                continue

            # 按文件名数字排序（1.bmp, 2.bmp, ..., 6.bmp）
            def bmp_key(path):
                name = os.path.basename(path)
                num = name.split('.')[0]  # 如 "1", "10"
                try:
                    return int(num)
                except:
                    return float('inf')
            bmp_files = sorted(bmp_files, key=bmp_key)

            # 为每个 BMP 文件添加一个样本（暂不启用翻转增强）
            for bmp_file in bmp_files:
                self.samples.append({'bmp': bmp_file, 'csv': csv_path})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # 加载图像（灰度图）
        img = Image.open(sample['bmp']).convert('L')
        img = np.array(img, dtype=np.uint8)  # [0, 255], uint8

        # 可选：亮度增强
        brightness_factor = random.choice([1.0, 1.0, 1.0])
        img = cv2.convertScaleAbs(img, alpha=brightness_factor)

        # 加载 depth map 并 ×10
        df = pd.read_csv(sample['csv'], header=None)
        depth_map = df.values.astype(np.float32) * 1.0

        # 应用 transform（ToTensor → [0,1]）
        img_tensor = self.transform(img)  # shape: (1, H, W)

        depth_tensor = torch.from_numpy(np.ascontiguousarray(depth_map)).unsqueeze(0).float()

        return {
            'image': img_tensor,
            'depth': depth_tensor
        }