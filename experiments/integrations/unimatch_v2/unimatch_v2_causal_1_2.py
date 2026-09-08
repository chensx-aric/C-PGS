import argparse
import math
from copy import deepcopy
import logging
import os
import pprint

import cv2
import numpy as np
import torch
from PIL import Image
from matplotlib.patches import Patch
from torch import nn
import torch.backends.cudnn as cudnn
from torch.optim import AdamW
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import yaml
from tqdm import tqdm

from dataset.semi import SemiDataset, SemiDatasetAdapted, SemiSupervisedSegDataset
from model.cls_model.clsmodel import CasualHyperGraph
from model.semseg.dpt import DPT
from supervised import evaluate
from util.classes import CLASSES
from util.ohem import ProbOhemCrossEntropy2d
from util.utils import count_params, init_log, AverageMeter
from scipy.ndimage import label as cc_label
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Set, Optional
parser = argparse.ArgumentParser(description='UniMatch V2: Pushing the Limit of Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local_rank', '--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
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
# python unimatch_v2_causal1.py --config ./configs/AC.yaml --labeled-id-path ./data --unlabeled-id-path ./data --save-path ./result5


def get_lambda_reg(epoch, total_epochs):
    # Sigmoid式后期增强
    return 1 / (1 + np.exp(-10 * (epoch / total_epochs - 0.5)))
def predict_single_image(model, image_path, label_mask, cfg,
                         mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
    """
    对单张图片进行语义分割预测并显示结果（带类别图例）
    """
    model.eval()

    # === 1. 读取图片 ===
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    orig_h, orig_w = img.shape[:2]

    # === 2. 调整为14的倍数 ===
    patch_size = 14
    new_h = math.ceil(orig_h / patch_size) * patch_size
    new_w = math.ceil(orig_w / patch_size) * patch_size
    img_resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # === 3. 归一化并转tensor ===
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    mean = torch.tensor(mean).view(1, 3, 1, 1)
    std = torch.tensor(std).view(1, 3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    img_tensor = img_tensor.cuda()

    # === 4. 模型预测 ===
    with torch.no_grad():
        pred = model(img_tensor)
        pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
        pred = pred.softmax(dim=1)
        pred_mask = pred.argmax(dim=1).squeeze()  # CUDA tensor [H,W]

        # label_mask originally numpy array
        label_mask_tensor = torch.from_numpy(label_mask).to(pred_mask.device)

        pred_mask = refine_with_sam(pred_mask, label_mask_tensor)
        pred_mask = pred_mask.cpu().numpy().astype(np.uint8)
    # with torch.no_grad():
    #     pred = model(img_tensor)
    #
    #     pred = F.interpolate(pred, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
    #     pred = pred.softmax(dim=1)
    #     pred_mask = pred.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
        # pred = refine_with_sam(pred_mask, label_mask)
        # top2_values, top2_indices = torch.topk(pred, k=3, dim=1)  # [B, 2, H, W]
        #
        # # 当前的 top1 和 top2 类别
        # top1_mask = top2_indices[:, 0, :, :]  # [B, H, W]
        # top2_mask = top2_indices[:, 1, :, :]
        #
        # # 替换条件：当 top1 为 17 时，用 top2 替代
        # replaced_mask = torch.where(top1_mask == 17, top2_mask, top1_mask)
        #
        # # 转 numpy
        # pred_mask = replaced_mask.squeeze().cpu().numpy().astype(np.uint8)

    # === 5. 生成彩色mask ===
    nclass = cfg.get('nclass', 18)
    color_mask = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
    for i in range(min(nclass, len(cfg['colors']))):
        color_mask[pred_mask == i] = cfg['colors'][i]
    overlay = (0.6 * img + 0.4 * color_mask).astype(np.uint8)
    class_names = cfg1['class_names']
    # === 6. 类别名设置 ===
    if class_names is None:
        class_names = [f"Class {i}" for i in range(nclass)]
    class_names = class_names[:nclass]

    # === 7. 生成图例 ===
    # === 7. 仅保留出现的类别 ===
    present_classes = np.unique(pred_mask)
    legend_elements = [
        Patch(facecolor=np.array(cfg['colors'][i]) / 255.0, label=class_names[i])
        for i in present_classes if i < nclass
    ]

    # === 8. 可视化 ===
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(color_mask)
    axes[1].set_title("Predicted Mask")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title("Overlay Result")
    axes[2].axis("off")

    # 图例放在右侧
    plt.legend(
        handles=legend_elements,
        loc='center left',
        bbox_to_anchor=(1.05, 0.5),
        title="Classes",
        fontsize=10,
    )

    plt.tight_layout()
    plt.show()

    return pred_mask, color_mask, overlay

def colorize_mask(mask, colors):
    """将类别 mask 转为彩色图像"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for i, color in enumerate(colors):
        color_mask[mask == i] = color
    return Image.fromarray(color_mask)



def get_lambda_prior_cls(epoch):
    if epoch < 20:
        return 0.9
    elif epoch < 40:
        return 0.0
    elif epoch < 60:
        return 0.0
    else:
        return 0.0
def get_affinity_weight(cur_iter, max_iter, start_ratio=0.15):
    # 前 30% 训练不启用该损失
    start_iter = int(max_iter * start_ratio)
    if cur_iter < start_iter:
        return 0.0
    t = (cur_iter - start_iter) / (max_iter - start_iter)
    t = max(0.0, min(1.0, t))
    return float(torch.exp(torch.tensor(-5 * (1 - t)**2)))
def main():
    args = parser.parse_args()
    device = torch.device("cuda:4") 
    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0


    # rank, world_size = setup_distributed(port=args.port)
    rank, world_size = 0, 1

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
    model = DPT(**{**model_configs[cfg['backbone'].split('_')[-1]], 'nclass': cfg['nclass'], 'device': device})
    state_dict = torch.load(f'./pretrained/{cfg["backbone"]}.pth')
    model.backbone.load_state_dict(state_dict)
    # ppp_fusion = PPPFusion(num_classes=18)
    # ppp_fusion.cuda()
    if cfg['lock_backbone']:
        model.lock_backbone()
    
    optimizer = AdamW(
        [
            {'params': [p for p in model.backbone.parameters() if p.requires_grad], 'lr': cfg['lr']},
            {'params': [param for name, param in model.named_parameters() if 'backbone' not in name], 'lr': cfg['lr'] * cfg['lr_multi']},
            # {'params': [p for p in ppp_fusion.parameters() if p.requires_grad], 'lr': cfg['lr'] * cfg['lr_multi']}
            # PPP params
        ], 
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=0.01
    )
    
    if rank == 0:
        logger.info('Total params: {:.1f}M'.format(count_params(model)))
        logger.info('Encoder params: {:.1f}M'.format(count_params(model.backbone)))
        logger.info('Decoder params: {:.1f}M\n'.format(count_params(model.head)))
    
    # local_rank = int(os.environ["LOCAL_RANK"])
    local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", 0))
    # clsmodel = CasualHyperGraph(256, 128, 18)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.to(device)
    # clsmodel.cuda()
    # checkpoint = torch.load("./model/simple_cls.pth", map_location="cuda")
    # clsmodel.load_state_dict(checkpoint["model_state"])

    # model = torch.nn.parallel.DistributedDataParallel(
    #     model, device_ids=[local_rank], broadcast_buffers=False, output_device=local_rank, find_unused_parameters=True
    # )
    
    model_ema = deepcopy(model)
    model_ema.eval()
    for param in model_ema.parameters():
        param.requires_grad = False
    
    if cfg['criterion']['name'] == 'CELoss':
        criterion_l = nn.CrossEntropyLoss(**cfg['criterion']['kwargs']).to(device)
    elif cfg['criterion']['name'] == 'OHEM':
        criterion_l = ProbOhemCrossEntropy2d(**cfg['criterion']['kwargs']).to(device)
    else:
        raise NotImplementedError('%s criterion is not implemented' % cfg['criterion']['name'])

    criterion_u = nn.CrossEntropyLoss(reduction='none').to(device)
    trainset_u = SemiDatasetAdapted(
        cfg['dataset'], 'train_u', '0.5',cfg['crop_size']
    )
    trainset_l = SemiDatasetAdapted(
        cfg['dataset'], 'train_l', '0.5', cfg['crop_size'], nsample=len(trainset_u.ids)
    )
    valset = SemiDatasetAdapted(
        cfg['dataset'], 'val', cfg['crop_size']
    )
    # trainset_u = SemiDataset(
    #     cfg['dataset'], cfg['data_root'], 'train_u', cfg['crop_size'], args.unlabeled_id_path
    # )
    # trainset_l = SemiDataset(
    #     cfg['dataset'], cfg['data_root'], 'train_l', cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids)
    # )
    # valset = SemiDataset(
    #     cfg['dataset'], cfg['data_root'], 'val'
    # )
    
    # trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    # trainloader_l = DataLoader(
    #     trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_l
    # )
    trainloader_l = DataLoader(
            trainset_l, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True
        )
    # trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    # trainloader_u = DataLoader(
    #     trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler_u
    # )
    trainloader_u = DataLoader(
        trainset_u, batch_size=cfg['batch_size'], pin_memory=True, num_workers=1, drop_last=True
    )
    
    # valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    # valloader = DataLoader(
    #     valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False, sampler=valsampler
    # )
    valloader = DataLoader(
        valset, batch_size=1, pin_memory=True, num_workers=1, drop_last=False
    )
    # proto_dataset = SemiSupervisedSegDataset(subset_name='labeled_5_new',
    #                                          image_features_path='./datasets/image_feature_new.pt', mode='use')
    # proto_loader = DataLoader(proto_dataset, batch_size=1, shuffle=False)




    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best, previous_best_ema = 0.0, 0.0
    best_epoch, best_epoch_ema = 0, 0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'), map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        previous_best_ema = checkpoint['previous_best_ema']
        best_epoch = checkpoint['best_epoch']
        best_epoch_ema = checkpoint['best_epoch_ema']

        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    # checkpoint = torch.load(os.path.join(args.save_path, 'best.pth'), map_location='cpu')
    # model.load_state_dict(checkpoint['model'])
    # # label_mask = np.load('./100067.npy').astype(np.uint8)
    # # pred_mask, color_mask, overlay = predict_single_image(model, '100067.jpg', label_mask, cfg1)
    # mIoU, iou_class = evaluate(model, valloader, 'original', cfg, device, multiplier=14)
    # for (cls_idx, iou) in enumerate(iou_class):
    #     logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
    #                 .format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou))
    # logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}\n'.format('original', mIoU))
    # lambda_cls0 = 0.5  # 初值
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, Previous best: {:.2f} @epoch-{:}, '
                        'EMA: {:.2f} @epoch-{:}'.format(epoch, previous_best, best_epoch, previous_best_ema, best_epoch_ema))
        
        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()

        # total_loss_r = AverageMeter()

        total_mask_ratio = AverageMeter()
        # if epoch < 30:
        #     aff_w = 0
        # elif epoch < 70:
        #     aff_w = 0.2
        # else:
        #     aff_w = 0
        # aff_w = get_affinity_weight(epoch, 200)

        # lambda_cls = lambda_cls0 * (1 - epoch / cfg['epochs'])
        # trainloader_l.sampler.set_epoch(epoch)
        # trainloader_u.sampler.set_epoch(epoch)
        # lambda_cls = get_lambda_prior_cls(epoch)
        loader = zip(trainloader_l, trainloader_u)
        
        model.train()
        total, matched = 0, 0
        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, ignore_mask, instance_mask, mask, cutmix_box1, cutmix_box2, cls_prior)) in enumerate(loader):
            
            img_x, mask_x = img_x.to(device), mask_x.to(device)
            img_u_w, img_u_s1, img_u_s2, instance_mask = img_u_w.to(device), img_u_s1.to(device), img_u_s2.to(device), instance_mask.to(device)
            ignore_mask, cutmix_box1, cutmix_box2 = ignore_mask.to(device), cutmix_box1.to(device), cutmix_box2.to(device)
            # for b in range(mask.size(0)):
            #     total += 1
            #
            #     # 获取 mask 中的类别集合
            #     cls_in_mask = torch.unique(mask[b])
            #     cls_in_mask = cls_in_mask[cls_in_mask != 255]  # 如果有 ignore label
            #
            #     # 获取 cls_prior 中包含的类别
            #     prior_classes = (cls_prior[b] > 0).nonzero(as_tuple=True)[0]
            #
            #     # 判断是否全部包含
            #     if set(cls_in_mask.tolist()).issubset(set(prior_classes.tolist())):
            #         matched += 1
            # mask = mask.cuda()
            with torch.no_grad():
                pred_u_w = model_ema(img_u_w).detach()
                conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
                mask_u_w = pred_u_w.argmax(dim=1)
            # with torch.no_grad():
            #     pred_u_w = model_ema(img_u_w).detach()  # [B, C, H, W]
            #     p_u_w = torch.softmax(pred_u_w, dim=1)  # softmax 概率
            #
            #     # ---- α 动态调整（置信度越低 α 越高）----
            #     conf_mean = p_u_w.max(dim=1)[0].mean()
            #     alpha = 0.5 * (1 - conf_mean)  # α ∈ [0, 0.5]
            #
            #     # ---- 构造先验 mask ----
            #     prior_mask = torch.zeros_like(p_u_w)
            #     prior_mask[:, known_classes, :, :] = 1.0
            #
            #     # ---- 增强 softmax 概率 ----
            #     p_u_w = p_u_w + alpha * prior_mask
            #     p_u_w = p_u_w / p_u_w.sum(dim=1, keepdim=True)
            #
            #     # ---- 根据增强后的概率重新取伪标签 ----
            #     conf_u_w, mask_u_w = p_u_w.max(dim=1)
            # with torch.no_grad():
            #     pred_u_w = model_ema(img_u_w).detach()  # [B, C, H, W]
            #     p_u_w = torch.softmax(pred_u_w, dim=1)  # [B, C, H, W]
            #
            #     # ---- α 动态调整 ----
            #     conf_mean = p_u_w.max(dim=1)[0].mean()
            #     alpha = 0.5 * (1 - conf_mean)
            #
            #     # ---- 构造逐样本的 prior mask ----
            #     B, C, H, W = p_u_w.shape
            #     prior_mask = torch.zeros_like(p_u_w)  # [B, C, H, W]
            #
            #     for b in range(B):
            #         # 🔥 提取当前图片的类别
            #         known_classes_b = cls_prior[b]
            #
            #         known_classes_b = torch.nonzero(known_classes_b).squeeze(1).tolist()
            #         # print("预测：{}".format(pred_u_w.argmax(dim=1)[b].unique()))
            #         # print("先验：{}".format(known_classes_b))
            #         if len(known_classes_b) > 0:
            #             prior_mask[b, known_classes_b, :, :] = 1.0
            #
            #     # ---- 增强 softmax 概率 ----
            #         p_u_w = p_u_w + alpha * prior_mask
            #     p_u_w = p_u_w / p_u_w.sum(dim=1, keepdim=True)
            #
            #     # ---- 新伪标签 ----
            #     conf_u_w, mask_u_w = p_u_w.max(dim=1)
            # 最好的时候用的这个
            # with torch.no_grad():
            #     pred_u_w = model_ema(img_u_w).detach()  # [B, C, H, W]
            #     p_model = torch.softmax(pred_u_w, dim=1)  # 模型预测概率
            #
            #     # ---- 计算模型平均置信度 ----
            #     conf_mean = p_model.max(dim=1)[0].mean()
            #
            #     # ---- prior 权重（前期高，后期低）----
            #     # 你可以调整 k（0.5~5 之间常用）
            #     k = 5.0
            #     w = torch.exp(-k * conf_mean).clamp(0, 1)
            #     if epoch % 10 == 0 and i == 0:
            #         print("模型置信度：{}".format(1-w))
            #     # ---- 生成 prior mask ----
            #     B, C, H, W = p_model.shape
            #     prior_mask = torch.zeros_like(p_model)
            #
            #     for b in range(B):
            #
            #         known_classes_b = torch.nonzero(cls_prior[b]).squeeze(1).tolist()
            #
            #         # print("预测：{}".format(pred_u_w.argmax(dim=1)[b].unique()))
            #         # print("先验：{}".format(known_classes_b))
            #
            #         if len(known_classes_b) > 0:
            #             prior_mask[b, known_classes_b, :, :] = 1.0
            #
            #     # ---- 将 prior_mask 归一化为 prior 概率 ----
            #     p_prior = prior_mask / (prior_mask.sum(dim=1, keepdim=True) + 1e-8)
            #
            #     # ---- 最终融合（关键改动）----
            #     p_u_w = (1 - w) * p_model + w * p_prior
            #
            #     # ---- 再归一化（确保概率合法）----
            #     p_u_w = p_u_w / p_u_w.sum(dim=1, keepdim=True)
            #     conf_u_w, mask_u_w = p_u_w.max(dim=1)

            # with torch.no_grad():
            #     pred_u_w = model_ema(img_u_w).detach()  # [B,C,H,W]
            #     p_model = torch.softmax(pred_u_w, dim=1)
            #     # conf_mean = p_model.max(dim=1)[0].mean()
            #     # k = 5.0
            #     # w = torch.exp(-k * conf_mean).clamp(0, 1)
            #     # if epoch % 10 == 0 and i == 0:
            #     #     print("模型置信度：{}".format(1 - w))
            #
            #     B, C, H, W = p_model.shape
            #     prior_mask = torch.zeros_like(p_model)
            #     for b in range(B):
            #         known_classes_b = torch.nonzero(cls_prior[b]).squeeze(1)
            #         if known_classes_b.numel() > 0:
            #             prior_mask[b, known_classes_b, :, :] = 1.0
            #
            #     p_prior = prior_mask / (prior_mask.sum(dim=1, keepdim=True) + 1e-8)
            #     # optionally normalize per pixel already done
            #
            # # ----- exit no_grad: now run PPP fusion (ppp_fusion has grad) -----
            # if epoch <= 30:
            #     pred_u_w = ppp_fusion(p_model, p_prior)  # [B,C,H,W]  <-- parameters of ppp_fusion will get grads
            #     conf_u_w = pred_u_w.max(dim=1)[0]
            #     mask_u_w = pred_u_w.argmax(dim=1)
            #     L_ppp = compute_ppp_loss_with_prior(p_model, p_prior, pred_u_w, epoch)
            # else:
            #     conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            #     mask_u_w = pred_u_w.argmax(dim=1)
            #     L_ppp = 0

            # with torch.no_grad():
            #     # ----- Teacher prediction -----
            #     pred_u_w = model_ema(img_u_w).detach()  # [B,C,H,W]
            #     p_model = torch.softmax(pred_u_w, dim=1)  # posterior
            #     conf_u_w, mask_u_w = p_model.max(dim=1)  # 原本你就需要

            #     B, C, H, W = p_model.shape

            #     # ----- Construct prior mask -----
            #     prior_mask = torch.zeros_like(p_model)  # [B,C,H,W]
            #     for b in range(B):
            #         known_classes_b = torch.nonzero(cls_prior[b]).squeeze(1)
            #         if known_classes_b.numel() > 0:
            #             prior_mask[b, known_classes_b, :, :] = 1.0

            # pixel_allowed = prior_mask.gather(1, mask_u_w.unsqueeze(1)).squeeze(1)
            # if epoch <= 10:
            #     ignore_prior = torch.zeros_like(mask_u_w)
            #     ignore_prior[(pixel_allowed == 0) & (conf_u_w < 0.6)] = 255
            # else:
            #     ignore_prior = torch.zeros_like(mask_u_w)
            #     ignore_prior[(pixel_allowed == 0) & (conf_u_w < 0.3)] = 255
            # ignore_mask = torch.where(
            #     (ignore_mask == 255) | (ignore_prior == 255),
            #     torch.tensor(255, device=ignore_mask.device),
            #     torch.tensor(0, device=ignore_mask.device)
            # )
            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = img_u_s1.flip(0)[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = img_u_s2.flip(0)[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]
            
            pred_x = model(img_x)
            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2)), comp_drop=True).chunk(2)


            mask_u_w_cutmixed1, conf_u_w_cutmixed1, ignore_mask_cutmixed1 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2, ignore_mask_cutmixed2 = mask_u_w.clone(), conf_u_w.clone(), ignore_mask.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w.flip(0)[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w.flip(0)[cutmix_box1 == 1]
            ignore_mask_cutmixed1[cutmix_box1 == 1] = ignore_mask.flip(0)[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w.flip(0)[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w.flip(0)[cutmix_box2 == 1]
            ignore_mask_cutmixed2[cutmix_box2 == 1] = ignore_mask.flip(0)[cutmix_box2 == 1]
            
            loss_x = criterion_l(pred_x, mask_x)

            loss_u_s1 = criterion_u(pred_u_s1, mask_u_w_cutmixed1)
            loss_u_s1 = loss_u_s1 * ((conf_u_w_cutmixed1 >= cfg['conf_thresh']) & (ignore_mask_cutmixed1 != 255))
            loss_u_s1 = loss_u_s1.sum() / (ignore_mask_cutmixed1 != 255).sum().item()
            
            loss_u_s2 = criterion_u(pred_u_s2, mask_u_w_cutmixed2)
            loss_u_s2 = loss_u_s2 * ((conf_u_w_cutmixed2 >= cfg['conf_thresh']) & (ignore_mask_cutmixed2 != 255))
            loss_u_s2 = loss_u_s2.sum() / (ignore_mask_cutmixed2 != 255).sum().item()
            
            loss_u_s = (loss_u_s1 + loss_u_s2) / 2.0
            # print("lambda_reg:{} lambda_cls:{}".format(lambda_reg, lambda_cls))
            # loss = (loss_x + loss_u_s) / 2.0 + loss_reg * lambda_reg + lambda_cls * loss_prior_cls
            loss = (loss_x + loss_u_s) / 2.0 #+ 0.1 * L_ppp

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update(loss_u_s.item())
            # total_loss_r.update(loss_reg.item())
            mask_ratio = ((conf_u_w >= cfg['conf_thresh']) & (ignore_mask != 255)).sum().item() / (ignore_mask != 255).sum()
            total_mask_ratio.update(mask_ratio.item())

            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            optimizer.param_groups[1]["lr"] = lr * cfg['lr_multi']
            
            ema_ratio = min(1 - 1 / (iters + 1), 0.996)
            
            for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                param_ema.copy_(param_ema * ema_ratio + param.detach() * (1 - ema_ratio))
            for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                buffer_ema.copy_(buffer_ema * ema_ratio + buffer.detach() * (1 - ema_ratio))
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', loss_u_s.item(), iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)

            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, LR: {:.7f}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, optimizer.param_groups[0]['lr'], total_loss.avg, total_loss_x.avg,
                                            total_loss_s.avg, total_mask_ratio.avg))
        # print(f"匹配图片数 / 总图片数 = {matched} / {total}")
        # print(f"匹配率: {matched / total:.4f}")
        eval_mode = 'sliding_window' if cfg['dataset'] == 'cityscapes' else 'original'
        mIoU, iou_class = evaluate(model, valloader, eval_mode, cfg, device, multiplier=14)
        mIoU_ema, iou_class_ema = evaluate(model_ema, valloader, eval_mode, cfg, device, multiplier=14)
        
        if rank == 0:
            for (cls_idx, iou) in enumerate(iou_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] IoU: {:.2f}, '
                            'EMA: {:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], iou, iou_class_ema[cls_idx]))
            logger.info('***** Evaluation {} ***** >>>> MeanIoU: {:.2f}, EMA: {:.2f}\n'.format(eval_mode, mIoU, mIoU_ema))
            
            writer.add_scalar('eval/mIoU', mIoU, epoch)
            writer.add_scalar('eval/mIoU_ema', mIoU_ema, epoch)
            for i, iou in enumerate(iou_class):
                writer.add_scalar('eval/%s_IoU' % (CLASSES[cfg['dataset']][i]), iou, epoch)
                writer.add_scalar('eval/%s_IoU_ema' % (CLASSES[cfg['dataset']][i]), iou_class_ema[i], epoch)

        is_best = mIoU >= previous_best
        
        previous_best = max(mIoU, previous_best)
        previous_best_ema = max(mIoU_ema, previous_best_ema)
        if mIoU == previous_best:
            best_epoch = epoch
        if mIoU_ema == previous_best_ema:
            best_epoch_ema = epoch
        
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'model_ema': model_ema.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'previous_best_ema': previous_best_ema,
                'best_epoch': best_epoch,
                'best_epoch_ema': best_epoch_ema
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
