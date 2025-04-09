import json
import os
import numpy as np
import torch
import nibabel as nib
import random
import matplotlib.pyplot as plt
class TrainSet(torch.utils.data.Dataset):
    def __init__(self, data_list, prefix='../../data/sd0809/BraTS2020', transform=None):
        json_path = 'brats20.json'
        data_list = self.load_json(json_path)
        self.data_list = data_list['train']
        self.prefix = prefix
        self.transform = transform

        # 加载所有数据
        self.images = []
        self.labels = []
        self.slices = []
        i = 0
        for data_item in self.data_list:
            images = [nib.load(os.path.join(self.prefix, image_path)).get_fdata()[:, :, 79:129] for image_path in data_item['image']]
            label = nib.load(os.path.join(self.prefix, data_item['label'])).get_fdata()[:, :, 79:129]
            
            # 确保所有模态和标签的形状一致
            assert all(image.shape == label.shape for image in images)
            
            # 体积归一化
            images = [self.normalize_volume(image) for image in images]
            
            self.images.append(images)
            self.labels.append(label)
            
            for slice_idx in range(50):  # 切片索引从0到49
                self.slices.append((i, slice_idx))
            print(data_item['label'])
            i += 1
            if i==30:
                break
        random.shuffle(self.slices)

    def __len__(self):
        return len(self.slices)

    def __getitem__(self, idx):
        img_idx, slice_idx = self.slices[idx]
        images = [img[:, :, slice_idx] for img in self.images[img_idx]]
        label = self.labels[img_idx][:, :, slice_idx]

        # Stack images along the channel dimension
        slice_images = np.stack(images, axis=0)

        # Optionally apply transformations
        if self.transform:
            slice_images, slice_label = self.transform(slice_images, label)

        # Convert to torch tensors
        slice_images = torch.tensor(slice_images, dtype=torch.float32).permute(0,2,1)
        slice_label = torch.tensor(label, dtype=torch.long).permute(1,0)
        #import pdb;pdb.set_trace()
        slice_label[slice_label == 4] = 3

        return slice_images, slice_label

    def normalize_volume(self, volume):
        """归一化体积数据到 [0, 1]"""
        volume = volume.astype(np.float32)
        min_val = np.min(volume)
        max_val = np.max(volume)
        normalized_volume = (volume - min_val) / (max_val - min_val + 1e-5)  # 加上 1e-5 以防止除零
        return normalized_volume

    def load_json(self, json_path):
        with open(json_path, 'r') as f:
            data = json.load(f)
        return data

def save_sample_images(dataset, idx):
    sample_images, sample_label = dataset[idx]

    # 将张量转换为 NumPy 数组以便于绘图
    sample_images = sample_images.numpy()
    sample_label = sample_label.numpy()

    fig, axes = plt.subplots(1, 5, figsize=(15, 3))

    for i in range(4):
        axes[i].imshow(sample_images[i], cmap='gray')
        axes[i].set_title(f'Modality {i+1}')
        axes[i].axis('off')

    axes[4].imshow(sample_label, cmap='gray')
    axes[4].set_title('Mask')
    axes[4].axis('off')

    plt.tight_layout()
    plt.savefig('sample_modalities_and_mask.png')
    plt.show()

if __name__ == "__main__":
    # 假设你有一个 brats20.json 文件和图像文件夹路径
    data_list = 'brats20.json'
    prefix = '../../data/sd0809/BraTS2020'

    # 创建数据集
    dataset = TrainSet(data_list, prefix)

    # 保存第一个样本的图像和标签
    save_sample_images(dataset, idx=0)
