import json

from dataset.transform import *

from copy import deepcopy
import math
import numpy as np
import os
from pathlib import Path
import random

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class SemiDataset(Dataset):
    def __init__(self, name, root, mode, size=None, id_path=None, nsample=None):
        self.name = name
        self.root = root
        self.mode = mode
        self.size = size
        
        if mode == 'train_l' or mode == 'train_u':
            with open(id_path, 'r') as f:
                self.ids = f.read().splitlines()
            if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
                self.ids *= math.ceil(nsample / len(self.ids))
                self.ids = self.ids[:nsample]
        else:
            with open('splits/%s/val.txt' % name, 'r') as f:
                self.ids = f.read().splitlines()

    def __getitem__(self, item):
        id = self.ids[item]
        img = Image.open(os.path.join(self.root, id.split(' ')[0])).convert('RGB')
        if self.mode == 'train_u':
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
        else:
            mask = Image.fromarray(np.array(Image.open(os.path.join(self.root, id.split(' ')[1])))) 
        
        if self.mode == 'val':
            img, mask = normalize(img, mask)
            return img, mask, id

        img, mask = resize(img, mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask = crop(img, mask, self.size, ignore_value)
        img, mask = hflip(img, mask, p=0.5)

        if self.mode == 'train_l':
            return normalize(img, mask)
        
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, cutmix_box1, cutmix_box2

    def __len__(self):
        return len(self.ids)
class SemiDatasetAdapted(Dataset):
        # def __init__(self, name, mode, data, size=None, nsample=None):
        #     """
        #     Args:
        #         name: 数据集名称（对应 ./datasets/{name}.txt）
        #         mode: train_l, train_u, val
        #         size: 裁剪尺寸
        #         nsample: 样本重复数量（用于小样本扩增）
        #     """
        #     self.name = name
        #     self.mode = mode
        #     self.size = size
        #     if mode == 'train_u':
        #         list_file = os.path.join('./datasets', f"train_unlabeled_{data}.txt")
        #         # self.prior_dict = torch.load(f"./datasets/cached_class_mask{data}.pt")
        #         self.prior_dict = torch.load(f"./datasets/cached_class_mask{data}nor.pt")
        #     elif mode == 'train_l':
        #         list_file = os.path.join('./datasets', f"train_labeled_{data}.txt")
        #     else:
        #         list_file = os.path.join('./datasets', f"val.txt")
        #     assert os.path.exists(list_file), f"{list_file} 不存在，请检查路径"

        #     with open(list_file, 'r') as f:
        #         self.ids = f.read().splitlines()
            
        #     # self.prior_dict = torch.load("./datasets/cached_class_mask1_16_nor.pt")
        #     # 小样本扩充
        #     if mode == 'train_l' and nsample is not None and nsample > len(self.ids):
        #         self.ids *= math.ceil(nsample / len(self.ids))
        #         self.ids = self.ids[:nsample]

        #     print(f"[{mode}] 共加载 {len(self.ids)} 个样本，来自 {list_file}")
    def __init__(self, name, mode, data, size=None, nsample=None):
        """
        Args:
            name: 数据集名称（对应 ./datasets/{name}.txt）
            mode: train_l, train_u, val
            size: 裁剪尺寸
            nsample: 固定样本长度（用于对齐迭代次数；train_l/train_u 都可用）
        """
        self.name = name
        self.mode = mode
        self.size = size

        repository_root = Path(__file__).resolve().parents[4]
        split_root = Path(
            os.environ.get('ACA_SPLIT_ROOT', repository_root / 'data' / 'splits' / 'aca')
        )

        if mode == 'train_u':
            list_file = split_root / f"train_unlabeled_{data}.txt"
            self.prior_dict = torch.load(f"./datasets/cached_class_mask{data}.pt")
            # self.prior_dict = torch.load(f"./datasets/cached_class_mask{data}nor.pt")
        elif mode == 'train_l':
            list_file = split_root / f"train_labeled_{data}.txt"
        else:
            list_file = split_root / "val.txt"

        assert os.path.exists(list_file), f"{list_file} 不存在，请检查路径"

        with open(list_file, 'r') as f:
            self.ids = f.read().splitlines()

        # =========================
        # ✅ 固定长度对齐（不随机）
        # train_l / train_u 都支持
        # =========================
        if mode in ['train_l', 'train_u'] and nsample is not None:
            n = len(self.ids)
            if n > 0 and nsample != n:
                if nsample > n:
                    # 1) 重复扩充
                    rep = math.ceil(nsample / n)
                    base_idx = list(range(n)) * rep
                    base_idx = base_idx[:nsample]
                else:
                    # 2) 确定性均匀下采样（避免只取前nsample造成偏置）
                    if nsample == 1:
                        base_idx = [0]
                    else:
                        base_idx = [round(i * (n - 1) / (nsample - 1)) for i in range(nsample)]

                # ids 同步
                self.ids = [self.ids[i] for i in base_idx]

                # ✅ prior_dict 同步（仅 train_u 需要）
                if mode == 'train_u' and hasattr(self, "prior_dict") and self.prior_dict is not None:
                    pd = self.prior_dict
                    # 情况A：prior_dict 是按行对齐的序列/张量（长度==原始n），则同步索引
                    if torch.is_tensor(pd) and pd.shape[0] == n:
                        idx_t = torch.tensor(base_idx, dtype=torch.long, device=pd.device)
                        self.prior_dict = pd.index_select(0, idx_t)
                    elif isinstance(pd, (list, tuple)) and len(pd) == n:
                        self.prior_dict = [pd[i] for i in base_idx]
                    # 情况B：prior_dict 是 dict（key=样本id/路径等），无需裁剪；重复样本也可正常索引
                    # elif isinstance(pd, dict):
                    #     pass

        print(f"[{mode}] 共加载 {len(self.ids)} 个样本，来自 {list_file}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        line = self.ids[item].strip().split()

        if len(line) != 2:
            raise ValueError(f"数据格式错误: {line}")
        repository_root = Path(__file__).resolve().parents[4]
        dataset_root = Path(
            os.environ.get('ACA_DATA_ROOT', repository_root / 'data' / 'ACA')
        )
        img_path = dataset_root / Path(line[0].replace("\\", "/"))
        label_path = dataset_root / Path(line[1].replace("\\", "/"))
        instance_label_path = Path(
            str(label_path).replace('SegmentationClassNpy', 'InstanceClassNpy')
        )
        # ---- 加载图像 ----
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train_u':
            cls_prior = self.prior_dict[item]
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
            # mask = Image.fromarray(np.load(label_path).astype(np.uint8))
        else:
            if label_path.suffix.lower() == '.npy':
                mask = Image.fromarray(np.load(label_path).astype(np.uint8))
            else:
                mask = Image.open(label_path)
        instance_mask = np.load(instance_label_path).astype(np.uint16)

        # ---- 验证模式 ----
        if self.mode == 'val':

            img, mask = normalize(img, mask)
            return img, mask, instance_mask, os.path.basename(img_path)
        instance_mask = Image.fromarray(instance_mask)
        # ---- 训练增强 ----
        img, mask, instance_mask = resize(img, mask, instance_mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask, instance_mask = crop(img, mask, self.size, ignore_value, instance_mask)
        img, mask, instance_mask = hflip(img, mask, 0.5, instance_mask)

        # ---- 有标签训练 ----
        if self.mode == 'train_l':
            return normalize(img, mask)

        # ---- 无标签训练 ----
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        # strong aug1
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        # strong aug2
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        instance_mask = torch.from_numpy(np.array(instance_mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, instance_mask, mask, cutmix_box1, cutmix_box2, cls_prior


class SemiDatasetAdapted1(Dataset):
    def __init__(self, name, mode, data, size=None, nsample=None):
        """
        Args:
            name: 数据集名称（对应 ./datasets/{name}.txt）
            mode: train_l, train_u, val
            size: 裁剪尺寸
            nsample: 固定样本长度（用于对齐迭代次数；train_l/train_u 都可用）
        """
        self.name = name
        self.mode = mode
        self.size = size

        repository_root = Path(__file__).resolve().parents[4]
        split_root = Path(
            os.environ.get('ACA_SPLIT_ROOT', repository_root / 'data' / 'splits' / 'aca')
        )

        if mode == 'train_u':
            list_file = split_root / f"train_unlabeled_{data}.txt"
            # self.prior_dict = torch.load(f"./datasets/cached_class_mask{data}.pt")
            self.prior_dict = torch.load(f"./datasets/cached_class_mask{data}nor.pt")
        elif mode == 'train_l':
            list_file = split_root / f"train_labeled_{data}.txt"
        else:
            list_file = split_root / "val.txt"

        assert os.path.exists(list_file), f"{list_file} 不存在，请检查路径"

        with open(list_file, 'r') as f:
            self.ids = f.read().splitlines()

        # =========================
        # ✅ 固定长度对齐（不随机）
        # train_l / train_u 都支持
        # =========================
        if mode in ['train_l', 'train_u'] and nsample is not None:
            n = len(self.ids)
            if n > 0 and nsample != n:
                if nsample > n:
                    # 1) 重复扩充
                    rep = math.ceil(nsample / n)
                    base_idx = list(range(n)) * rep
                    base_idx = base_idx[:nsample]
                else:
                    # 2) 确定性均匀下采样（避免只取前nsample造成偏置）
                    if nsample == 1:
                        base_idx = [0]
                    else:
                        base_idx = [round(i * (n - 1) / (nsample - 1)) for i in range(nsample)]

                # ids 同步
                self.ids = [self.ids[i] for i in base_idx]

                # ✅ prior_dict 同步（仅 train_u 需要）
                if mode == 'train_u' and hasattr(self, "prior_dict") and self.prior_dict is not None:
                    pd = self.prior_dict
                    # 情况A：prior_dict 是按行对齐的序列/张量（长度==原始n），则同步索引
                    if torch.is_tensor(pd) and pd.shape[0] == n:
                        idx_t = torch.tensor(base_idx, dtype=torch.long, device=pd.device)
                        self.prior_dict = pd.index_select(0, idx_t)
                    elif isinstance(pd, (list, tuple)) and len(pd) == n:
                        self.prior_dict = [pd[i] for i in base_idx]
                    # 情况B：prior_dict 是 dict（key=样本id/路径等），无需裁剪；重复样本也可正常索引
                    # elif isinstance(pd, dict):
                    #     pass

        print(f"[{mode}] 共加载 {len(self.ids)} 个样本，来自 {list_file}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, item):
        line = self.ids[item].strip().split()

        if len(line) != 2:
            raise ValueError(f"数据格式错误: {line}")
        repository_root = Path(__file__).resolve().parents[4]
        dataset_root = Path(
            os.environ.get('ACA_DATA_ROOT', repository_root / 'data' / 'ACA')
        )
        img_path = dataset_root / Path(line[0].replace("\\", "/"))
        label_path = dataset_root / Path(line[1].replace("\\", "/"))
        instance_label_path = Path(
            str(label_path).replace('SegmentationClassNpy', 'InstanceClassNpy')
        )
        # ---- 加载图像 ----
        img = Image.open(img_path).convert('RGB')
        if self.mode == 'train_u':
            cls_prior = self.prior_dict[item]
            mask = Image.fromarray(np.zeros((img.size[1], img.size[0]), dtype=np.uint8))
            # mask = Image.fromarray(np.load(label_path).astype(np.uint8))
        else:
            if label_path.suffix.lower() == '.npy':
                mask = Image.fromarray(np.load(label_path).astype(np.uint8))
            else:
                mask = Image.open(label_path)
        instance_mask = np.load(instance_label_path).astype(np.uint16)

        # ---- 验证模式 ----
        if self.mode == 'val':

            img, mask = normalize(img, mask)
            return img, mask, instance_mask, os.path.basename(img_path)
        instance_mask = Image.fromarray(instance_mask)
        # ---- 训练增强 ----
        img, mask, instance_mask = resize(img, mask, instance_mask, (0.5, 2.0))
        ignore_value = 254 if self.mode == 'train_u' else 255
        img, mask, instance_mask = crop(img, mask, self.size, ignore_value, instance_mask)
        img, mask, instance_mask = hflip(img, mask, 0.5, instance_mask)

        # ---- 有标签训练 ----
        if self.mode == 'train_l':
            return normalize(img, mask)

        # ---- 无标签训练 ----
        img_w, img_s1, img_s2 = deepcopy(img), deepcopy(img), deepcopy(img)

        # strong aug1
        if random.random() < 0.8:
            img_s1 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s1)
        img_s1 = transforms.RandomGrayscale(p=0.2)(img_s1)
        img_s1 = blur(img_s1, p=0.5)
        cutmix_box1 = obtain_cutmix_box(img_s1.size[0], p=0.5)

        # strong aug2
        if random.random() < 0.8:
            img_s2 = transforms.ColorJitter(0.5, 0.5, 0.5, 0.25)(img_s2)
        img_s2 = transforms.RandomGrayscale(p=0.2)(img_s2)
        img_s2 = blur(img_s2, p=0.5)
        cutmix_box2 = obtain_cutmix_box(img_s2.size[0], p=0.5)

        ignore_mask = Image.fromarray(np.zeros((mask.size[1], mask.size[0])))

        img_s1, ignore_mask = normalize(img_s1, ignore_mask)
        img_s2 = normalize(img_s2)

        mask = torch.from_numpy(np.array(mask)).long()
        instance_mask = torch.from_numpy(np.array(instance_mask)).long()
        ignore_mask[mask == 254] = 255

        return normalize(img_w), img_s1, img_s2, ignore_mask, instance_mask, mask, cutmix_box1, cutmix_box2, cls_prior
