import os
import json
import random
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms

def split_dataset_for_semi(
    class_image_map_path: str,
    output_dir: str = "./datasets",
    val_ratio: float = 0.10,
    ratios: List[float] = (1/16, 1/8, 1/4, 1/2),
    seed: int = 42,
):
    """
    基于 class_image_map 的划分函数（方案 B，满足你所有要求）。

    Args:
        class_image_map_path: JSON 文件路径，格式为 { "0": [[img, mask], ...], "1": [...], ... }
        output_dir: 保存所有 txt 的目录
        val_ratio: 验证集占比（默认 0.10）
        ratios: 半监督 labeled 比例（基于 train_full）
        seed: 随机种子，便于复现

    Returns:
        dict: 包含生成的集合（路径对）的字典
    """

    random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    # ----------------------------
    # 1) 读入 class_image_map 并构建映射
    # ----------------------------
    with open(class_image_map_path, 'r', encoding='utf-8') as f:
        class_to_images = json.load(f)

    # 确定类别数（从 keys 推断）
    class_ids = sorted([int(k) for k in class_to_images.keys()])
    n_classes = max(class_ids) + 1 if class_ids else 0

    # image -> set(classes)
    image_to_classes: Dict[str, Set[int]] = defaultdict(set)
    # image -> style
    image_to_style: Dict[str, str] = {}
    # image -> one example label path (first encountered)
    image_to_label: Dict[str, str] = {}

    for cls_str, pairs in class_to_images.items():
        cls = int(cls_str)
        for img_path_raw, label_path in pairs:
            img_path = img_path_raw.replace("/", "\\")
            # normalize path stray whitespace
            img_path = img_path.strip()
            image_to_classes[img_path].add(cls)
            if img_path not in image_to_label:
                image_to_label[img_path] = label_path.replace("/", "\\").strip()
            # style extraction:倒数第3层
            parts = img_path.split("\\")
            if len(parts) >= 3:
                style = parts[-3]
            else:
                style = "UNKNOWN_STYLE"
            image_to_style[img_path] = style

    all_images = sorted(list(image_to_classes.keys()))
    total_images = len(all_images)
    if total_images == 0:
        raise ValueError("class_image_map 中没有图像条目。")

    # 构建 style -> class -> list(images)
    style_to_class_images: Dict[str, Dict[int, List[str]]] = defaultdict(lambda: defaultdict(list))
    for img in all_images:
        style = image_to_style[img]
        for cls in image_to_classes[img]:
            style_to_class_images[style][cls].append(img)

    styles = sorted(style_to_class_images.keys())

    # ----------------------------
    # 2) 确保 train/val 都覆盖每个 style-class（尽最大努力）
    #    - 若某 style-class 唯一一张图 -> 放 train（并发警告）
    #    - 若 >=2 -> 分配不同图到 train & val（若同图被多个类共享，会尽力避免冲突）
    # ----------------------------
    forced_train: Set[str] = set()
    forced_val: Set[str] = set()
    used_train: Set[str] = set()
    used_val: Set[str] = set()

    # 按稀缺度处理：先处理样本少的 (style, cls) 对，减少冲突
    style_class_pairs = []
    for style in styles:
        for cls, imgs in style_to_class_images[style].items():
            style_class_pairs.append((len(imgs), style, cls))
    style_class_pairs.sort(key=lambda x: x[0])  # 从少到多

    for count, style, cls in style_class_pairs:
        imgs = style_to_class_images[style][cls][:]
        random.shuffle(imgs)

        if count == 0:
            # 如果没有任何图片（罕见），跳过
            print(f"[警告] 无样本：style={style}, class={cls}")
            continue

        if count == 1:
            # 只有一张图片，只能放入 train（无法保证 val 同时覆盖）
            img = imgs[0]
            if img in used_val:
                # 如果被占用在 val（因为它也满足别的 style-class），我们 prefer 保持 val 的覆盖，
                # 这里仍然将其加入 train 可能导致 train/val 重复 -> 我们避免重复：将另一张可替代图片放 train（若有）
                # 但因为 count==1，说明没有可替代 -> 直接加入 train（且输出警告）
                pass
            forced_train.add(img)
            used_train.add(img)
            # 警告
            print(f"[WARN] Only 1 image for style={style}, class={cls}: assigned to train (val cannot contain this pair).")
            continue

        # count >= 2: 需要为 val & train 各挑一张不同图片
        # 选择 val_img: 首先尝试选一个未被使用过的图片
        val_img = None
        for img in imgs:
            if (img not in used_val) and (img not in used_train):
                val_img = img
                break
        # 如果没有找到干净的，允许选未在 val 中使用但可能在 train 中使用的（尽力避免重复）
        if val_img is None:
            for img in imgs:
                if img not in used_val:
                    val_img = img
                    break
        # 选择 train_img: 选择与 val_img 不同且未被使用的
        train_img = None
        for img in imgs:
            if img == val_img:
                continue
            if (img not in used_train) and (img not in used_val):
                train_img = img
                break
        if train_img is None:
            for img in imgs:
                if img != val_img and img not in used_train:
                    train_img = img
                    break

        # 最终保障
        if val_img is None:
            # 退化：把第一个放 val（可能已被其他分配占用）
            val_img = imgs[0]
            print(f"[WARN] cannot find clean val_img for ({style},{cls}), using {val_img}")

        if train_img is None:
            # 退化：选择 imgs 中除了 val_img 的任意一张（可能冲突）
            for img in imgs:
                if img != val_img:
                    train_img = img
                    break
            if train_img is None:
                # 极端情况：所有 imgs 都相同（其实不可能），把 val_img 也当作 train（我们避免重复 later）
                train_img = val_img

        # assign
        forced_val.add(val_img)
        forced_train.add(train_img)
        used_val.add(val_img)
        used_train.add(train_img)

    # ----------------------------
    # 3) 用剩余图片补齐 train/val 到目标比例（90/10）
    #    并确保 train/val 互不重复
    # ----------------------------
    target_val = int(round(total_images * val_ratio))
    target_train = total_images - target_val

    remaining = [img for img in all_images if img not in used_train and img not in used_val]
    random.shuffle(remaining)

    val_set = set(forced_val)
    train_set = set(forced_train)

    # 先确保 val 不超过目标
    for img in remaining[:]:
        if len(val_set) >= target_val:
            break
        # prefer images that help increase style/class coverage in val (optional)
        val_set.add(img)
        remaining.remove(img)

    # 其余全部进入 train（确保互不重复）
    for img in remaining:
        if img not in val_set:
            train_set.add(img)

    # 如果 val 太少（极端），可以从 train 中移动一些不影响强制覆盖的图片
    if len(val_set) < target_val:
        need = target_val - len(val_set)
        movable = [img for img in list(train_set) if img not in forced_train]
        if len(movable) < need:
            print(f"[WARN] 无足够可移动样本将 train 补到 val({need})，当前 movable={len(movable)}")
            # 尽可能移动
        # move first `need` movables
        for img in movable[:need]:
            train_set.remove(img)
            val_set.add(img)

    # 如果 train 太小或太大，调整（一般不会）
    if len(train_set) != target_train:
        # 尝试调整以精确匹配 target_train
        diff = len(train_set) - target_train
        if diff > 0:
            # train 太多，搬一些到 val（优先非强制 train）
            candidates = [img for img in train_set if img not in forced_train]
            move_num = min(diff, len(candidates))
            for img in candidates[:move_num]:
                train_set.remove(img)
                val_set.add(img)
        elif diff < 0:
            # train 太少，从 val 挪一些非强制 val 到 train
            need = -diff
            candidates = [img for img in val_set if img not in forced_val]
            move_num = min(need, len(candidates))
            for img in candidates[:move_num]:
                val_set.remove(img)
                train_set.add(img)

    # 最终检查
    if len(train_set) != target_train or len(val_set) != target_val:
        print(f"[WARN] 最终 train/val 大小与目标不符: train={len(train_set)}({target_train}), val={len(val_set)}({target_val})")

    print(f"[Step1完成] total={total_images}, train_full={len(train_set)}, val={len(val_set)}")

    # ----------------------------
    # 4) 在 train_full 上生成严格数量的 labeled/unlabeled（每个比例）
    #    每个 labeled 尽量包含 train 中每个 style-class 的至少 1 张（若 cover > target，则随机截断）
    # ----------------------------
    train_list = sorted(list(train_set))
    train_size = len(train_list)

    def build_min_cover_from_train(train_imgs: List[str]) -> Set[str]:
        """从 train_imgs 中为每个 (style, class) 取至少 1 张作为覆盖集（若可用）。"""
        cover = set()
        for style in styles:
            for cls in style_to_class_images[style].keys():
                imgs = [img for img in style_to_class_images[style][cls] if img in train_set]
                if imgs:
                    cover.add(random.choice(imgs))
        return cover

    results = {
        "train_full": train_list,
        "val": sorted(list(val_set)),
    }

    # helper: write image+label pairs
    def save_pairs(image_list: List[str], out_path: str):
        with open(out_path, "w", encoding="utf-8") as f:
            for img in sorted(image_list):
                label = image_to_label.get(img, "")
                f.write(f"{img} {label}\n")

    # save train_full & val
    save_pairs(results["train_full"], os.path.join(output_dir, "train_full.txt"))
    save_pairs(results["val"], os.path.join(output_dir, "val.txt"))

    # generate for each ratio
    for r in ratios:
        target_num = max(1, int(round(train_size * r)))

        # minimal cover
        cover_set = build_min_cover_from_train(train_list)
        cover_count = len(cover_set)

        if cover_count > target_num:
            # cover 太大：随机截断（会导致某些 style-class 可能不被覆盖）
            print(f"[WARN] minimal cover ({cover_count}) > target ({target_num}) for ratio {r}. Truncating cover randomly.")
            cover_list = list(cover_set)
            random.shuffle(cover_list)
            labeled_set = set(cover_list[:target_num])
        else:
            # cover <= target: 补齐
            labeled_set = set(cover_set)
            remaining_candidates = [img for img in train_list if img not in labeled_set]
            random.shuffle(remaining_candidates)
            need = target_num - len(labeled_set)
            labeled_set.update(remaining_candidates[:need])

        unlabeled_set = set(train_list) - labeled_set

        # safety checks
        assert len(labeled_set) == target_num, f"labeled size mismatch for ratio {r}"
        assert len(unlabeled_set) == train_size - target_num, f"unlabeled size mismatch for ratio {r}"

        # save
        r_str = str(r).replace("/", "_")
        labeled_name = os.path.join(output_dir, f"train_labeled_{r}.txt")
        unlabeled_name = os.path.join(output_dir, f"train_unlabeled_{r}.txt")

        save_pairs(sorted(list(labeled_set)), labeled_name)
        save_pairs(sorted(list(unlabeled_set)), unlabeled_name)

        results[f"labeled_{r}"] = sorted(list(labeled_set))
        results[f"unlabeled_{r}"] = sorted(list(unlabeled_set))

        print(f"[Ratio {r}] labeled={len(labeled_set)}, unlabeled={len(unlabeled_set)}")

    print("[全部划分完成] 输出目录：", os.path.abspath(output_dir))
    return results


class CustomDataset(Dataset):
    def __init__(self, file_list: str, target_size=(1024, 1024), transform=None, label_transform=None):
        """
        初始化 Dataset 类，用于读取图像和标签（包括实例标签）。

        Args:
            file_list: 训练/验证/测试集文件路径（每行包括图像路径和标签路径）。
            target_size: 图像和标签调整的目标大小，默认 (256, 256)。
            transform: 可选的图像变换函数。
            label_transform: 可选的标签变换函数。
        """
        self.file_list = self._load_file_list(file_list)
        self.target_size = target_size
        self.transform = transform
        self.label_transform = label_transform

        # 定义图像变换（如必要）
        self.image_transform = transforms.Compose([
            transforms.Resize(self.target_size),
            transforms.ToTensor(),
        ])

        # 定义标签变换（如必要）
        self.label_transform_func = transforms.Compose([
            transforms.Resize(self.target_size, interpolation=Image.NEAREST)
        ])

    def _load_file_list(self, file_list: str):
        """
        加载文件列表，每一行包含图像路径和标签路径。

        Args:
            file_list: txt 文件路径（每行格式为：图像路径 标签路径）。

        Returns:
            List[Tuple[str, str]]: [(image_path, label_path), ...]
        """
        with open(file_list, 'r') as f:
            file_pairs = f.readlines()

        # 移除每行结尾的空白字符，并拆分为 (image_path, label_path) 对
        file_pairs = [tuple(line.strip().split()) for line in file_pairs]
        return file_pairs

    def __len__(self):
        """返回数据集的大小"""
        return len(self.file_list)

    def __getitem__(self, idx):
        """根据索引返回一个样本（图像、标签、实例标签）"""
        img_path, label_path = self.file_list[idx]

        # 读取图像
        image = Image.open(img_path).convert('RGB')

        # 读取标签
        label = np.load(label_path)  # 假设标签是以 NumPy 数组的格式存储
        label = Image.fromarray(label)

        # 读取实例标签
        instance_label_path = label_path.replace('SegmentationClassNpy', 'InstanceClassNpy')
        instance_label = np.load(instance_label_path).astype(np.uint8)
        instance_label = Image.fromarray(instance_label)

        # 如果有变换（例如数据增强），则应用
        if self.transform:
            image = self.transform(image)
            label = self.label_transform_func(label)
            instance_label = self.label_transform_func(instance_label)

        # 处理图像（包括尺寸调整和转换为Tensor）
        image = self.image_transform(image)

        # 转换标签和实例标签为张量
        label = torch.from_numpy(np.array(label)).long()
        instance_label = torch.from_numpy(np.array(instance_label)).long()

        return image, label, instance_label

def save_image_features(train_loader, image_encoder, feature_save_dir, device='cuda'):
    """
    对每张图片进行特征提取并保存。

    Args:
        train_loader (DataLoader): 训练数据的 DataLoader。
        image_encoder (nn.Module): 图像编码器模型，用于提取图像特征。
        feature_save_dir (str): 特征保存的目录。
        device (str): 设备选择，'cuda' 或 'cpu'。

    保存特征到特定目录，每张图像保存为单独的 `.npy` 文件，文件名由图像文件名派生。
    """

    # 确保特征保存的目录存在
    os.makedirs(feature_save_dir, exist_ok=True)

    # 设置编码器为评估模式
    image_encoder.eval()

    # 遍历 train_loader 中的每个批次
    with torch.no_grad():
        for idx, (image, _) in tqdm(enumerate(train_loader), total=len(train_loader), desc="Extracting features"):
            # 将数据转移到指定设备
            image = image.to(device)

            # 通过模型提取特征
            # 使用 image.unsqueeze(0) 来增加一个批次维度
            output = image_encoder(image.unsqueeze(0))
            features = output["vision_features"]  # 获取 vision_features

            # 假设图像编码器输出的特征是 (batch_size, feature_dim)
            features = features.cpu().numpy()

            # 保存每张图片的特征
            for i in range(features.shape[0]):
                # 获取图片的原始文件名（例如：train_labeled_0.0625.txt 中的图像路径）
                img_path = train_loader.dataset.file_list[idx * train_loader.batch_size + i][0]
                img_name = os.path.basename(img_path).split('.')[0]  # 获取文件名，不包含扩展名

                # 保存每张图片的特征到.npy文件
                feature_file = os.path.join(feature_save_dir, f"{img_name}.npy")
                np.save(feature_file, features[i])

    print(f"Feature extraction completed. Features saved to {feature_save_dir}")

if __name__ == "__main__":
    split_dataset_for_semi('../datasets/class_image_map.json')
    train_dataset = CustomDataset(file_list='../dataset/datasets/train_full.txt', label_transform=None)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    # save_image_features(train_loader, image_encoder, '../dataset/datasets/', device='cuda')