import argparse
import logging
import glob
import os
import pprint

import time
import cv2
import torch
import numpy as np
from matplotlib import pyplot as plt
from torch import nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.semi import SemiDataset, SemiDatasetAdapted
from model.semseg.dpt import DPT
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, AverageMeter, intersectionAndUnion, init_log
from util.dist_helper import setup_distributed
from scipy.ndimage import label as cc_label

parser = argparse.ArgumentParser(description='Fully-Supervised Training in Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, default=None)
parser.add_argument('--pretrained-path', type=str, default=None)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--data', default='0.0625', type=str)
cfg1 = {
    'mean': [0.485, 0.456, 0.406],
    'std': [0.229, 0.224, 0.225],
    'nclass': 18,
    'colors': [
        [0, 0, 0],        # 0 background
        [70, 70, 70],     # 1 building
        [128, 64, 128],   # 2 road
        [244, 35, 232],   # 3 sidewalk
        [107, 142, 35],   # 4 vegetation
        [70, 130, 180],   # 5 sky
        [220, 20, 60],    # 6 person
        [0, 0, 142],      # 7 car
        [0, 60, 100],     # 8 bus
        [0, 80, 100],     # 9 truck
        [153, 153, 153],  # 10 pole
        [250, 170, 30],   # 11 traffic light
        [220, 220, 0],    # 12 sign
        [190, 153, 153],  # 13 fence
        [102, 102, 156],  # 14 wall
        [152, 251, 152],  # 15 terrain
        [255, 0, 0],      # 16 rider
        [119, 11, 32],    # 17 bicycle
    ],
    'class_names': ['bakground', 'architecture', 'stairs', 'window', 'pillar', 'tsun-shou', 'door',
                 'stone step', 'architrave', 'plaque', 'dou-gong', 'ridge', 'pagoda',
                 'stone lion', 'stone elephant', 'fence', 'censer', 'stone tablet']
}
def classify_sam_instances(pred_mask, sam_instances, dominance_ratio=0.6):
    """
    pred_mask:     [H, W] 模型预测类别图
    sam_instances: [H, W] SAM 实例 ID 图（0 为背景）
    dominance_ratio: 某类别达到该占比则认为是压倒性优势
    """

    H, W = pred_mask.shape
    sam_cls_mask = torch.zeros_like(pred_mask)

    inst_ids = torch.unique(sam_instances)
    # inst_ids = inst_ids[inst_ids != 0]  # 忽略背景

    for inst_id in inst_ids:
        region = (sam_instances == inst_id)
        region_pixels = pred_mask[region]

        # =============================
        # 1. Majority vote + 占比统计
        # =============================
        hist = torch.bincount(region_pixels)
        major_vote = hist.argmax().item()
        total_pixels = region_pixels.numel()
        major_ratio = hist[major_vote].item() / total_pixels

        # =============================
        # 2. 中心点类别
        # =============================
        ys, xs = region.nonzero(as_tuple=True)
        cy = (ys.min() + ys.max()) // 2
        cx = (xs.min() + xs.max()) // 2
        center_cls = pred_mask[cy, cx].item()

        # =============================
        # 3. 检查是否存在非背景/建筑体类别
        # =============================
        unique_classes = torch.unique(region_pixels)
        non_base_classes = unique_classes[(unique_classes != 0) & (unique_classes != 1)]

        # =============================
        # 4. 决策逻辑（核心）
        # =============================
        if major_ratio >= dominance_ratio:
            # 若某类别的占比达到压倒性优势
            chosen_cls = major_vote

        else:
            # 否则根据非基础类别情况进行中心点优先策略
            if len(non_base_classes) > 0:
                # 若存在高语义类别但不占优势 → 中心点类别
                chosen_cls = center_cls
            else:
                # 若区域只有0/1 → 根据多数投票
                chosen_cls = major_vote

        # =============================
        # 5. 赋值
        # =============================
        sam_cls_mask[region] = chosen_cls

    return sam_cls_mask

# def fuse_prediction(
#         pred_mask, sam_cls_mask,
#         base_iou=0.3,
#         expand_pixels=60,
#         min_area_ratio=0.0002,    # 小于该比例忽略 SAM 区域 (~H*W*0.0002)
#         neighbor_consistency=0.25, # SAM区域周围pred同类比例 <0.25则不替换
#         category_cfg=None
#     ):
#     """
#     更鲁棒的融合策略:
#     pred_mask: [H,W]  模型预测
#     sam_cls_mask: [H,W]  SAM结果
#     """

#     if category_cfg is None:
#         category_cfg = {
#             4: {"force": True},           # 类别4直接信任SAM，可自行扩展更多类别规则
#         }

#     fused = pred_mask.clone()
#     H, W = pred_mask.shape
#     total_pixels = H * W

#     pred_np = pred_mask.cpu().numpy()
#     sam_np = sam_cls_mask.cpu().numpy()

#     classes = np.unique(sam_np)

#     for c in classes:
#         if c in [0,1]:
#             continue

#         cfg = category_cfg.get(c,{})

#         sam_c = (sam_np == c).astype(np.uint8)
#         num_labels, labels = cv2.connectedComponents(sam_c, connectivity=8)

#         for region_id in range(1, num_labels):

#             region_mask_np = (labels == region_id).astype(np.uint8)
#             area = region_mask_np.sum()

#             # ① 小区域过滤（减少误触发）
#             if area < total_pixels * min_area_ratio:
#                 continue

#             # ② 类别强制替换
#             if cfg.get("force",False):
#                 fused[torch.from_numpy(region_mask_np).to(pred_mask.device).bool()] = c
#                 continue

#             # ③ IoU + 动态阈值
#             pred_region_np = (pred_np == c).astype(np.uint8)
#             inter = np.logical_and(region_mask_np,pred_region_np).sum()
#             union = np.logical_or(region_mask_np,pred_region_np).sum()
#             iou = inter/(union+1e-6)

#             # 区域越小IoU宽松
#             region_ratio = area/total_pixels
#             iou_thr = base_iou - 0.25*np.exp(-6*region_ratio)  # 可调形状
#             if iou < iou_thr:
#                 continue

#             if inter == 0:
#                 continue

#             # ④ SAM & pred邻域一致性检查
#             kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(7,7))
#             neighbor = cv2.dilate(region_mask_np,kernel_small)-region_mask_np
#             if neighbor.sum()>0:
#                 nb_ratio = (pred_np[neighbor==1]==c).mean()
#                 if nb_ratio < neighbor_consistency:
#                     continue

#             # ⑤ 交集区域膨胀限制覆盖范围
#             intersection_np = (region_mask_np & pred_region_np)
#             kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(expand_pixels*2+1,expand_pixels*2+1))
#             allowed_np = cv2.dilate(intersection_np.astype(np.uint8),kernel,iterations=1)

#             final_region_np = region_mask_np & allowed_np
#             final_region = torch.from_numpy(final_region_np).to(pred_mask.device).bool()
#             fused[final_region] = c

#     return fused
def fuse_prediction(
        pred_mask, sam_cls_mask,
        base_iou=0.3,
        expand_pixels=60,
        min_area_ratio=0.0002,     # 小于该比例忽略 SAM 区域 (~H*W*0.0002)
        neighbor_consistency=0.25, # SAM区域周围pred同类比例 <0.25则不替换
        category_cfg=None
    ):
    """
    更鲁棒的融合策略:
    pred_mask: [H,W]  模型预测
    sam_cls_mask: [H,W]  SAM结果

    category_cfg:
      - {"force": True}    : 该类别完全信任SAM，直接替换
      - {"pred_only": True}: 该类别完全信任pred，不使用SAM去修改
    """

    if category_cfg is None:
        category_cfg = {
            4: {"force": True},      # 类别4直接信任SAM
            # 例如：2: {"pred_only": True},  # 类别2完全用模型预测，不让SAM改
            6: {"pred_only": True},
            10: {"pred_only": True},
            11: {"pred_only": True},
            13: {"pred_only": True},
            16: {"pred_only": True},
        }

    fused = pred_mask.clone()
    H, W = pred_mask.shape
    total_pixels = H * W

    pred_np = pred_mask.cpu().numpy()
    sam_np = sam_cls_mask.cpu().numpy()

    classes = np.unique(sam_np)

    for c in classes:
        if c in [0, 1]:
            continue

        cfg = category_cfg.get(c, {})

        # ✅ 新增：pred_only 类别直接跳过（保持 fused=pred_mask 的结果）
        if cfg.get("pred_only", False):
            continue

        sam_c = (sam_np == c).astype(np.uint8)
        num_labels, labels = cv2.connectedComponents(sam_c, connectivity=8)

        for region_id in range(1, num_labels):
            region_mask_np = (labels == region_id).astype(np.uint8)
            area = region_mask_np.sum()

            # ① 小区域过滤（减少误触发）
            if area < total_pixels * min_area_ratio:
                continue

            # ② 类别强制替换（信任SAM）
            if cfg.get("force", False):
                fused[torch.from_numpy(region_mask_np).to(pred_mask.device).bool()] = c
                continue

            # ③ IoU + 动态阈值
            pred_region_np = (pred_np == c).astype(np.uint8)
            inter = np.logical_and(region_mask_np, pred_region_np).sum()
            union = np.logical_or(region_mask_np, pred_region_np).sum()
            iou = inter / (union + 1e-6)

            region_ratio = area / total_pixels
            iou_thr = base_iou - 0.25 * np.exp(-6 * region_ratio)
            if iou < iou_thr:
                continue

            if inter == 0:
                continue

            # ④ 邻域一致性检查
            kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            neighbor = cv2.dilate(region_mask_np, kernel_small) - region_mask_np
            if neighbor.sum() > 0:
                nb_ratio = (pred_np[neighbor == 1] == c).mean()
                if nb_ratio < neighbor_consistency:
                    continue

            # ⑤ 交集区域膨胀限制覆盖范围
            intersection_np = (region_mask_np & pred_region_np)
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (expand_pixels * 2 + 1, expand_pixels * 2 + 1)
            )
            allowed_np = cv2.dilate(intersection_np.astype(np.uint8), kernel, iterations=1)

            final_region_np = region_mask_np & allowed_np
            final_region = torch.from_numpy(final_region_np).to(pred_mask.device).bool()
            fused[final_region] = c

    return fused

# # ---------------------------------------------------------------
# Step 3：彩色可视化工具
# ---------------------------------------------------------------
def colorize(mask):
    mask_np = mask.cpu().numpy().astype(np.uint8)
    color_mask = palette=np.array(cfg1['colors'], dtype=np.uint8)[mask_np]
    return color_mask

def visualize_three(pred_mask, sam_cls_mask, fused_mask):

    pred_color = colorize(pred_mask)
    sam_color = colorize(sam_cls_mask)
    fused_color = colorize(fused_mask)

    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.title("Model Prediction")
    plt.imshow(pred_color)
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("SAM Classified Instances")
    plt.imshow(sam_color)
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Fused Result")
    plt.imshow(fused_color)
    plt.axis("off")

    plt.show()

# ---------------------------------------------------------------
# 调用示例
# pred_mask      -> 模型预测 [H, W]
# sam_instances  -> SAM2 自动掩码器输出的实例掩码 [H, W]
# ---------------------------------------------------------------
import imageio
def save_fused_only(fused_mask, save_path):
    fused_color = colorize(fused_mask)  # HxWx3

    # 确保是 uint8（0~255），避免保存发灰/不正常
    if fused_color.dtype != np.uint8:
        fused_color = np.clip(fused_color, 0, 255).astype(np.uint8)

    imageio.imwrite(save_path, fused_color)
    return fused_color
# def refine_with_sam(pred_mask, sam_instances):

#     # 1. 对 SAM 实例进行分类
#     sam_cls_mask = classify_sam_instances(pred_mask, sam_instances)

#     # 2. 融合结果
#     fused_mask = fuse_prediction(pred_mask, sam_cls_mask)

#     # 3. 彩色展示
#     # visualize_three(pred_mask, sam_cls_mask, fused_mask)

#     return fused_mask
def refine_with_sam(pred_mask, sam_instances=None, id=0):

    # 1. 对 SAM 实例进行分类
    if sam_instances != None:
        sam_cls_mask = classify_sam_instances(pred_mask, sam_instances)

        # 2. 融合结果
        fused_mask = fuse_prediction(pred_mask, sam_cls_mask)

    # 3. 彩色展示
    # visualize_three(pred_mask, sam_cls_mask, fused_mask)
    # fused_color = visualize_fused_only(fused_mask)
    else:
        fused_mask = pred_mask
    save_fused_only(fused_mask, f"image_result2/{id}.png")
    return fused_mask


# def evaluate(model, loader, mode, cfg, multiplier=None):
#     model.eval()
#     assert mode in ['original', 'center_crop', 'sliding_window']
#     intersection_meter = AverageMeter()
#     union_meter = AverageMeter()

#     with torch.no_grad():
#         for img, mask, instance_mask, id in loader:
            
#             img = img.cuda()
#             instance_mask = instance_mask.cuda()
#             if mode == 'sliding_window':
#                 grid = cfg['crop_size']
#                 b, _, h, w = img.shape
#                 final = torch.zeros(b, 19, h, w).cuda()
                
#                 row = 0
#                 while row < h:
#                     col = 0
#                     while col < w:
#                         pred = model(img[:, :, row: row + grid, col: col + grid])
#                         final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)
#                         if col == w - grid:
#                             break
#                         col = min(col + int(grid * 2 / 3), w - grid)
#                     if row == h - grid:
#                         break
#                     row = min(row + int(grid * 2 / 3), h - grid)
                    
#                 pred = final
            
#             else:
#                 assert mode == 'original'
                
#                 if multiplier is not None:
#                     ori_h, ori_w = img.shape[-2:]
#                     if multiplier == 512:
#                         new_h, new_w = 512, 512
#                     else:
#                         new_h, new_w = int(ori_h / multiplier + 0.5) * multiplier, int(ori_w / multiplier + 0.5) * multiplier
#                     img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)

#                 pred = model(img)

            
#                 if multiplier is not None:
#                     pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)
            
#             pred = pred.argmax(dim=1)
#             pred = refine_with_sam(pred[0], instance_mask[0], id).unsqueeze(0)
#             # pred = refine_with_sam(pred[0], id=id).unsqueeze(0)
#             intersection, union, target = \
#                 intersectionAndUnion(pred.cpu().numpy(), mask.numpy(), cfg['nclass'], 255)

#             reduced_intersection = torch.from_numpy(intersection).cuda()
#             reduced_union = torch.from_numpy(union).cuda()
#             reduced_target = torch.from_numpy(target).cuda()

#             if dist.is_initialized():
#                 dist.all_reduce(reduced_intersection)
#                 dist.all_reduce(reduced_union)
#                 dist.all_reduce(reduced_target)

#             intersection_meter.update(reduced_intersection.cpu().numpy())
#             union_meter.update(reduced_union.cpu().numpy())

#     iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
#     mIOU = np.mean(iou_class)

#     return mIOU, iou_class
def evaluate(model, loader, mode, cfg, multiplier=None, measure_time=False, warmup=5):
    model.eval()
    assert mode in ['original', 'center_crop', 'sliding_window']
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()

    model_times = []
    refine_times = []
    total_times = []

    with torch.no_grad():
        for idx, (img, mask, instance_mask, id) in enumerate(loader):

            img = img.cuda()
            instance_mask = instance_mask.cuda()

            # =========================
            # 总推理计时开始
            # =========================
            if measure_time:
                torch.cuda.synchronize()
                total_start = time.time()

                torch.cuda.synchronize()
                model_start = time.time()

            # =========================
            # 1. Model prediction time
            # =========================
            if mode == 'sliding_window':
                grid = cfg['crop_size']
                b, _, h, w = img.shape
                final = torch.zeros(b, 19, h, w).cuda()

                row = 0
                while row < h:
                    col = 0
                    while col < w:
                        pred = model(img[:, :, row: row + grid, col: col + grid])
                        final[:, :, row: row + grid, col: col + grid] += pred.softmax(dim=1)

                        if col == w - grid:
                            break
                        col = min(col + int(grid * 2 / 3), w - grid)

                    if row == h - grid:
                        break
                    row = min(row + int(grid * 2 / 3), h - grid)

                pred = final

            else:
                assert mode == 'original'

                if multiplier is not None:
                    ori_h, ori_w = img.shape[-2:]
                    if multiplier == 512:
                        new_h, new_w = 512, 512
                    else:
                        new_h = int(ori_h / multiplier + 0.5) * multiplier
                        new_w = int(ori_w / multiplier + 0.5) * multiplier
                    img = F.interpolate(img, (new_h, new_w), mode='bilinear', align_corners=True)

                pred = model(img)

                if multiplier is not None:
                    pred = F.interpolate(pred, (ori_h, ori_w), mode='bilinear', align_corners=True)

            pred = pred.argmax(dim=1)

            if measure_time:
                torch.cuda.synchronize()
                model_end = time.time()

            # =========================
            # 2. refine_with_sam time
            # =========================
            if measure_time:
                torch.cuda.synchronize()
                refine_start = time.time()

            pred = refine_with_sam(pred[0], instance_mask[0], id).unsqueeze(0)

            if measure_time:
                torch.cuda.synchronize()
                refine_end = time.time()

                torch.cuda.synchronize()
                total_end = time.time()

                # 跳过前 warmup 张，避免初始化影响
                if idx >= warmup:
                    model_times.append(model_end - model_start)
                    refine_times.append(refine_end - refine_start)
                    total_times.append(total_end - total_start)

            # =========================
            # 原来的 mIoU 计算
            # =========================
            intersection, union, target = intersectionAndUnion(
                pred.cpu().numpy(),
                mask.numpy(),
                cfg['nclass'],
                255
            )

            reduced_intersection = torch.from_numpy(intersection).cuda()
            reduced_union = torch.from_numpy(union).cuda()
            reduced_target = torch.from_numpy(target).cuda()

            if dist.is_initialized():
                dist.all_reduce(reduced_intersection)
                dist.all_reduce(reduced_union)
                dist.all_reduce(reduced_target)

            intersection_meter.update(reduced_intersection.cpu().numpy())
            union_meter.update(reduced_union.cpu().numpy())

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10) * 100.0
    mIOU = np.mean(iou_class)

    if measure_time:
        avg_model_time = float(np.mean(model_times))
        avg_refine_time = float(np.mean(refine_times))
        avg_total_time = float(np.mean(total_times))

        print(f"[Time] Model prediction: {avg_model_time:.4f} s/image")
        print(f"[Time] refine_with_sam:  {avg_refine_time:.4f} s/image")
        print(f"[Time] Total inference:  {avg_total_time:.4f} s/image")

        return mIOU, iou_class, {
            "model_time": avg_model_time,
            "refine_time": avg_refine_time,
            "total_time": avg_total_time,
        }

    return mIOU, iou_class

def main():
    # CUDA_VISIBLE_DEVICES=6 python supervised.py --config ./configs/AC.yaml --labeled-id-path ./data --unlabeled-id-path ./data --save-path ./supervised1_2 --data 0.5
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO, os.path.join(args.save_path, 'train.log'))
    logger.propagate = 0

    # rank, world_size = setup_distributed(port=args.port)
    rank, world_size = 0, 1

    cfg['batch_size'] *= 2
    
    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model_configs = {
        'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass']})
    
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
    
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']}
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
    
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", 0))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    # )
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).cuda(local_rank)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).cuda(local_rank)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])
    
    n_upsampled = {
        'AC':3000,
        'pascal': 3000, 
        'cityscapes': 3000, 
        'ade20k': 6000, 
        'coco': 30000
    }
    # trainset = SemiDataset(
    #     cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=n_upsampled[cfg['dataset']]
    # )
    # valset = SemiDataset(
    #     cfg['dataset'], cfg['data_root'], 'val'
    # )
    trainset = SemiDatasetAdapted(
        cfg['dataset'], 'train_l', args.data, cfg['crop_size'], nsample=n_upsampled[cfg['dataset']]
    )
    valset = SemiDatasetAdapted(
        cfg['dataset'], 'val', args.data, cfg['crop_size']
    )
    
    # trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(
        trainset, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True
    )
    
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False
    )
    
    iters = 0
    total_iters = len(trainloader) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    # if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
    #     checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
    #     model.load_state_dict(checkpoint['model'])
    #     optimizer.load_state_dict(checkpoint['optimizer'])
    #     epoch = checkpoint['epoch']
    #     previous_best = checkpoint['previous_best']
        
    #     if rank == 0:
    #         logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
    model.load_state_dict(checkpoint['model'])
    mIoU, iou_class = evaluate(model, valloader, 'original', cfg, multiplier=14)
    for (cls_idx, iou) in enumerate(iou_class):
        logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                    .format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format('original', mIoU))
    # for epoch in range(epoch + 1, cfg['epochs']):
    #     if rank == 0:
    #         logger.info('===========> Epoch: {:}, LR: {:.7f}, Previous best: {:.2f}'.format(
    #             epoch, optimizer.param_groups[0]['lr'], previous_best))

    #     model.train()
    #     total_loss = AverageMeter()

    #     # trainsampler.set_epoch(epoch)

    #     for i, (img, mask) in enumerate(trainloader):

    #         img, mask = img.cuda(), mask.cuda()

    #         pred = model(img)

    #         loss = criterion(pred, mask)
            
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    #         total_loss.update(loss.item())

    #         iters = epoch * len(trainloader) + i
    #         lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
    #         optimizer.param_groups[0]["lr"] = lr
    #         optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
    #         if rank == 0:
    #             writer.add_scalar('train/loss_all', loss.item(), iters)
    #             writer.add_scalar('train/loss_x', loss.item(), iters)
            
    #         if (i % (len(trainloader) // 8) == 0) and (rank == 0):
    #             logger.info('Iters: {:}, Total loss: {:.3f}'.format(i, total_loss.avg))
        
    #     eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
    #     mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, multiplier=14)
        
    #     if rank == 0:
    #         for (cls_idx, iou) in enumerate(iou_class):
    #             logger.info('***** Evaluation ***** >>>> Class [{:} {:}] '
    #                         'IoU: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    #         logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format(eval_mode, mIoU))
            
    #         writer.add_scalar('eval/mIoU', mIoU, epoch)
    #         for i, iou in enumerate(iou_class):
    #             writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
        
    #     is_best = mIoU > previous_best
    #     previous_best = max(mIoU, previous_best)
    #     if rank == 0:
    #         checkpoint = {
    #             'model': model.state_dict(),
    #             'optimizer': optimizer.state_dict(),
    #             'epoch': epoch,
    #             'previous_best': previous_best,
    #         }
    #         torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
    #         if is_best:
    #             torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
