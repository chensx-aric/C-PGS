"""Train or validate the C-PGS UniMatch-V2 integration on ACA.

One entry point covers every labeled-data split.  The historical experiment
scripts differed mainly in hard-coded split names, CUDA device indices, and
inactive commented branches; those choices are now explicit command-line
arguments.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import random
from typing import Any, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader, DistributedSampler, Subset
import yaml


INTEGRATION_ROOT = Path(__file__).resolve().parent
REPOSITORY_ROOT = INTEGRATION_ROOT.parents[2]

SPLIT_FRACTIONS = {
    "1/16": "0.0625",
    "1/8": "0.125",
    "1/4": "0.25",
    "1/2": "0.5",
}
SPLIT_ALIASES = {
    "1_16": "1/16",
    "1_8": "1/8",
    "1_4": "1/4",
    "1_2": "1/2",
    **{fraction: name for name, fraction in SPLIT_FRACTIONS.items()},
}
EXPECTED_LABELED_COUNTS = {"1/16": 82, "1/8": 164, "1/4": 327, "1/2": 654}
EXPECTED_TRAIN_SIZE = 1308
EXPECTED_VALIDATION_SIZE = 145

MODEL_CONFIGS = {
    "small": {"encoder_size": "small", "features": 64, "out_channels": [48, 96, 192, 384]},
    "base": {"encoder_size": "base", "features": 128, "out_channels": [96, 192, 384, 768]},
    "large": {"encoder_size": "large", "features": 256, "out_channels": [256, 512, 1024, 1024]},
    "giant": {"encoder_size": "giant", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
}


@dataclass(frozen=True)
class ExperimentPaths:
    data_root: Path
    split_root: Path
    labeled_list: Path
    unlabeled_list: Path
    validation_list: Path
    prior_cache: Path | None
    pretrained: Path | None
    output_dir: Path


def canonical_split(value: str) -> str:
    """Return the paper-style split name for a supported alias."""

    normalized = value.strip()
    if normalized in SPLIT_FRACTIONS:
        return normalized
    try:
        return SPLIT_ALIASES[normalized]
    except KeyError as exc:
        choices = ", ".join(SPLIT_FRACTIONS)
        raise argparse.ArgumentTypeError(f"unsupported split {value!r}; choose one of {choices}") from exc


def _repo_path(value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPOSITORY_ROOT / path).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="C-PGS on UniMatch-V2 for all ACA labeled-data splits"
    )
    parser.add_argument("--split", type=canonical_split, default="1/2")
    parser.add_argument("--config", type=Path, default=INTEGRATION_ROOT / "configs" / "AC.yaml")
    parser.add_argument("--data-root", type=Path, help="ACA directory; overrides data_root in the config")
    parser.add_argument("--split-root", type=Path, help="directory containing the ACA split manifests")
    parser.add_argument("--labeled-list", type=Path, help="explicit labeled manifest override")
    parser.add_argument("--unlabeled-list", type=Path, help="explicit unlabeled manifest override")
    parser.add_argument("--validation-list", type=Path, help="explicit validation manifest override")
    parser.add_argument("--prior-cache", type=Path, help="class-level SP tensor/list for the unlabeled manifest")
    parser.add_argument("--pretrained", type=Path, help="DINOv2 backbone initialization weights")
    parser.add_argument("--output-dir", type=Path, help="default: outputs/unimatch_v2/<split>")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint used by --evaluate-only")
    parser.add_argument(
        "--checkpoint-key",
        choices=("model", "model_ema"),
        default="model_ema",
        help="state dictionary to evaluate (default: model_ema)",
    )
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--check-only", action="store_true", help="validate paths, manifests, and SP cache, then exit")
    parser.add_argument("--no-style-prior", action="store_true", help="disable Equation 16 for a baseline run")
    parser.add_argument("--alpha", type=float, help="style-prior logit strength")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, help="batch size per process/GPU")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--confidence-threshold", type=float)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", help="single-process device, e.g. cuda, cuda:0, or cpu")
    parser.add_argument("--port", type=int, help="legacy SLURM master-port override")
    parser.add_argument("--local-rank", "--local_rank", type=int, default=0)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--skip-validation", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, help=argparse.SUPPRESS)
    return parser


def resolve_paths(args: argparse.Namespace, cfg: Mapping[str, Any]) -> ExperimentPaths:
    fraction = SPLIT_FRACTIONS[args.split]
    split_root_value = args.split_root or cfg.get("split_root", "data/splits/aca")
    data_root_value = args.data_root or cfg.get("data_root", "data/ACA")
    split_root = _repo_path(split_root_value)
    data_root = _repo_path(data_root_value)

    labeled = _repo_path(args.labeled_list) if args.labeled_list else split_root / f"train_labeled_{fraction}.txt"
    unlabeled = (
        _repo_path(args.unlabeled_list)
        if args.unlabeled_list
        else split_root / f"train_unlabeled_{fraction}.txt"
    )
    validation = _repo_path(args.validation_list) if args.validation_list else split_root / "val.txt"

    prior: Path | None
    if args.no_style_prior:
        prior = None
    elif args.prior_cache:
        prior = _repo_path(args.prior_cache)
    else:
        prior = INTEGRATION_ROOT / "datasets" / f"cached_class_mask{fraction}.pt"

    pretrained_value = args.pretrained or cfg.get("pretrained_path")
    pretrained = _repo_path(pretrained_value) if pretrained_value else None

    output_value = args.output_dir or Path("outputs") / "unimatch_v2" / args.split.replace("/", "_")
    return ExperimentPaths(
        data_root=data_root,
        split_root=split_root,
        labeled_list=labeled.resolve(),
        unlabeled_list=unlabeled.resolve(),
        validation_list=validation.resolve(),
        prior_cache=prior.resolve() if prior else None,
        pretrained=pretrained,
        output_dir=_repo_path(output_value),
    )


def _read_manifest(path: Path) -> list[tuple[str, str]]:
    entries: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            fields = raw_line.strip().split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected '<image> <label>'")
            entries.append((fields[0], fields[1]))
    if not entries:
        raise ValueError(f"manifest is empty: {path}")
    return entries


def _load_torch(path: Path, *, weights_only: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=weights_only)
    except TypeError:  # PyTorch < 2.0 compatibility
        return torch.load(path, map_location="cpu")


def _validate_prior_cache(cache: Any, entries: list[tuple[str, str]], nclass: int) -> None:
    if isinstance(cache, Mapping):
        missing = [image for image, _ in entries if image not in cache and image.replace("\\", "/") not in cache]
        if missing:
            raise ValueError(f"style-prior mapping is missing {len(missing)} manifest entries; first: {missing[0]}")
        priors = [cache.get(image, cache.get(image.replace("\\", "/"))) for image, _ in entries]
    elif isinstance(cache, (list, tuple)) or torch.is_tensor(cache):
        if len(cache) != len(entries):
            raise ValueError(f"style-prior cache has {len(cache)} rows; manifest has {len(entries)}")
        priors = cache
    else:
        raise TypeError("style-prior cache must be a tensor, sequence, or image-path mapping")

    for index, prior in enumerate(priors):
        tensor = torch.as_tensor(prior)
        if tuple(tensor.shape) != (nclass,):
            raise ValueError(f"style-prior row {index} has shape {tuple(tensor.shape)}; expected ({nclass},)")
        if not bool(torch.all((tensor == 0) | (tensor == 1))):
            raise ValueError(f"style-prior row {index} is not binary")


def validate_experiment(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    paths: ExperimentPaths,
    *,
    check_data_files: bool,
) -> dict[str, Any]:
    nclass = int(cfg["nclass"])
    required = [paths.labeled_list, paths.unlabeled_list, paths.validation_list]
    can_resume = not args.no_resume and (paths.output_dir / "latest.pth").is_file()
    if not args.evaluate_only and not can_resume:
        required.append(paths.pretrained if paths.pretrained is not None else Path("<missing --pretrained>"))
    if not args.no_style_prior and not args.evaluate_only:
        if paths.prior_cache is None:
            required.append(Path("<missing --prior-cache>"))
        else:
            required.append(paths.prior_cache)
    if args.evaluate_only:
        if args.checkpoint is None:
            required.append(Path("<missing --checkpoint>"))
        else:
            required.append(_repo_path(args.checkpoint))
    if check_data_files:
        required.append(paths.data_root)

    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("required assets are missing:\n  " + "\n  ".join(missing))

    labeled = _read_manifest(paths.labeled_list)
    unlabeled = _read_manifest(paths.unlabeled_list)
    validation = _read_manifest(paths.validation_list)
    expected_labeled = EXPECTED_LABELED_COUNTS[args.split]
    if len(labeled) != expected_labeled:
        raise ValueError(f"{args.split} labeled manifest has {len(labeled)} rows; expected {expected_labeled}")
    if len(labeled) + len(unlabeled) != EXPECTED_TRAIN_SIZE:
        raise ValueError(
            f"labeled + unlabeled rows equal {len(labeled) + len(unlabeled)}; expected {EXPECTED_TRAIN_SIZE}"
        )
    if len(validation) != EXPECTED_VALIDATION_SIZE:
        raise ValueError(f"validation manifest has {len(validation)} rows; expected {EXPECTED_VALIDATION_SIZE}")

    if paths.prior_cache and not args.evaluate_only:
        cache = _load_torch(paths.prior_cache, weights_only=True)
        _validate_prior_cache(cache, unlabeled, nclass)

    if check_data_files:
        manifest_entries = (
            (paths.labeled_list, labeled),
            (paths.unlabeled_list, unlabeled),
            (paths.validation_list, validation),
        )
        for manifest, entries in manifest_entries:
            for image_rel, label_rel in entries:
                image_path = paths.data_root / Path(image_rel.replace("\\", "/"))
                if not image_path.is_file():
                    raise FileNotFoundError(f"image referenced by {manifest} is missing: {image_path}")
                if manifest != paths.unlabeled_list:
                    label_path = paths.data_root / Path(label_rel.replace("\\", "/"))
                    if not label_path.is_file():
                        raise FileNotFoundError(f"label referenced by {manifest} is missing: {label_path}")

    return {
        "split": args.split,
        "fraction": SPLIT_FRACTIONS[args.split],
        "labeled_images": len(labeled),
        "unlabeled_images": len(unlabeled),
        "validation_images": len(validation),
        "style_prior": not args.no_style_prior,
        "data_root": str(paths.data_root),
        "prior_cache": str(paths.prior_cache) if paths.prior_cache else None,
        "pretrained": str(paths.pretrained) if paths.pretrained else None,
        "output_dir": str(paths.output_dir),
    }


def _setup_runtime(args: argparse.Namespace) -> tuple[torch.device, int, int, bool]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    distributed = world_size > 1

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("distributed C-PGS training requires CUDA")
        local_rank = int(os.environ.get("LOCAL_RANK", str(args.local_rank)))
        torch.cuda.set_device(local_rank)
        if args.port is not None and "MASTER_PORT" not in os.environ:
            os.environ["MASTER_PORT"] = str(args.port)
        dist.init_process_group(backend="nccl", init_method="env://")
        device = torch.device("cuda", local_rank)
    else:
        requested = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(requested)
        if device.type == "cuda":
            device_index = torch.cuda.current_device() if device.index is None else device.index
            torch.cuda.set_device(device_index)
            device = torch.device("cuda", device_index)
    return device, rank, world_size, distributed


def _configure_logger(output_dir: Path, rank: int, *, write_file: bool = True) -> logging.Logger:
    logger = logging.getLogger("cpgs.unimatch_v2")
    logger.handlers.clear()
    logger.propagate = False
    logger.setLevel(logging.INFO)
    if rank != 0:
        logger.addHandler(logging.NullHandler())
        return logger

    formatter = logging.Formatter("[%(asctime)s][%(levelname)8s] %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    logger.addHandler(stream)
    if write_file:
        output_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(output_dir / "train.log", mode="a", encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def _seed_everything(seed: int, rank: int) -> None:
    value = seed + rank
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(value)


def _unwrap(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DistributedDataParallel) else model


def _normalise_state_dict(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    if state and all(key.startswith("module.") for key in state):
        return {key[7:]: value for key, value in state.items()}
    return dict(state)


def _select_state(payload: Any, key: str) -> Mapping[str, torch.Tensor]:
    if isinstance(payload, Mapping) and key in payload:
        state = payload[key]
    elif isinstance(payload, Mapping) and "state_dict" in payload:
        state = payload["state_dict"]
    else:
        state = payload
    if not isinstance(state, Mapping):
        raise TypeError(f"checkpoint entry {key!r} is not a state dictionary")
    return _normalise_state_dict(state)


def _build_model(cfg: Mapping[str, Any], device: torch.device) -> nn.Module:
    from model.semseg.dpt import DPT

    backbone_size = str(cfg["backbone"]).split("_")[-1]
    if backbone_size not in MODEL_CONFIGS:
        raise ValueError(f"unsupported DINOv2 backbone: {cfg['backbone']}")
    model = DPT(
        **MODEL_CONFIGS[backbone_size],
        nclass=int(cfg["nclass"]),
        device=device,
    )
    if bool(cfg.get("lock_backbone", False)):
        model.lock_backbone()
    return model.to(device)


def _load_backbone(model: nn.Module, path: Path) -> None:
    payload = _load_torch(path, weights_only=True)
    if isinstance(payload, Mapping) and "state_dict" in payload:
        payload = payload["state_dict"]
    if not isinstance(payload, Mapping):
        raise TypeError(f"pretrained backbone is not a state dictionary: {path}")
    model.backbone.load_state_dict(_normalise_state_dict(payload), strict=True)


def _make_optimizer(model: nn.Module, cfg: Mapping[str, Any]) -> AdamW:
    backbone = [parameter for parameter in model.backbone.parameters() if parameter.requires_grad]
    decoder = [parameter for name, parameter in model.named_parameters() if "backbone" not in name]
    return AdamW(
        [
            {"params": backbone, "lr": float(cfg["lr"])},
            {"params": decoder, "lr": float(cfg["lr"]) * float(cfg["lr_multi"])},
        ],
        lr=float(cfg["lr"]),
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )


def _make_criterion(cfg: Mapping[str, Any], device: torch.device) -> nn.Module:
    criterion_cfg = cfg["criterion"]
    if criterion_cfg["name"] != "CELoss":
        raise ValueError("the published ACA protocol requires criterion.name=CELoss")
    return nn.CrossEntropyLoss(**criterion_cfg["kwargs"]).to(device)


def _make_loaders(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    paths: ExperimentPaths,
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
) -> tuple[DataLoader, DataLoader, DataLoader, DistributedSampler | None, DistributedSampler | None]:
    from dataset.semi import ACASemiDataset

    train_u = ACASemiDataset(
        paths.data_root,
        paths.unlabeled_list,
        "train_u",
        crop_size=int(cfg["crop_size"]),
        num_classes=int(cfg["nclass"]),
        prior_path=paths.prior_cache,
    )
    train_l = ACASemiDataset(
        paths.data_root,
        paths.labeled_list,
        "train_l",
        crop_size=int(cfg["crop_size"]),
        num_classes=int(cfg["nclass"]),
        nsample=len(train_u),
    )
    validation = ACASemiDataset(
        paths.data_root,
        paths.validation_list,
        "val",
        num_classes=int(cfg["nclass"]),
    )

    sampler_l = DistributedSampler(train_l, shuffle=True, seed=args.seed) if distributed else None
    sampler_u = DistributedSampler(train_u, shuffle=True, seed=args.seed) if distributed else None
    if distributed:
        validation = Subset(validation, range(rank, len(validation), world_size))

    common = {
        "batch_size": int(cfg["batch_size"]),
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "drop_last": True,
    }
    loader_l = DataLoader(train_l, sampler=sampler_l, shuffle=sampler_l is None, **common)
    loader_u = DataLoader(train_u, sampler=sampler_u, shuffle=sampler_u is None, **common)
    loader_v = DataLoader(
        validation,
        batch_size=1,
        num_workers=min(args.workers, 2),
        pin_memory=device.type == "cuda",
        shuffle=False,
    )
    if len(loader_u) == 0:
        raise ValueError("unlabeled loader has zero batches; reduce --batch-size")
    return loader_l, loader_u, loader_v, sampler_l, sampler_u


def _make_validation_loader(
    args: argparse.Namespace,
    cfg: Mapping[str, Any],
    paths: ExperimentPaths,
    rank: int,
    world_size: int,
    distributed: bool,
    device: torch.device,
) -> DataLoader:
    from dataset.semi import ACASemiDataset

    validation: torch.utils.data.Dataset = ACASemiDataset(
        paths.data_root,
        paths.validation_list,
        "val",
        num_classes=int(cfg["nclass"]),
    )
    if distributed:
        validation = Subset(validation, range(rank, len(validation), world_size))
    return DataLoader(
        validation,
        batch_size=1,
        num_workers=min(args.workers, 2),
        pin_memory=device.type == "cuda",
        shuffle=False,
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    device: torch.device,
    num_classes: int,
    multiplier: int = 14,
    distributed: bool = False,
) -> tuple[float, np.ndarray]:
    model.eval()
    intersection = torch.zeros(num_classes, dtype=torch.float64, device=device)
    union = torch.zeros_like(intersection)

    for image, target, _sample_id in loader:
        image = image.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        original_size = image.shape[-2:]
        height = max(multiplier, int(original_size[0] / multiplier + 0.5) * multiplier)
        width = max(multiplier, int(original_size[1] / multiplier + 0.5) * multiplier)
        if (height, width) != original_size:
            image = F.interpolate(image, (height, width), mode="bilinear", align_corners=True)
        logits = model(image)
        if logits.shape[-2:] != original_size:
            logits = F.interpolate(logits, original_size, mode="bilinear", align_corners=True)
        prediction = logits.argmax(dim=1)

        valid = target != 255
        prediction = prediction[valid]
        target = target[valid]
        encoded = target * num_classes + prediction
        confusion = torch.bincount(encoded, minlength=num_classes * num_classes).reshape(num_classes, num_classes)
        intersection += confusion.diag()
        union += confusion.sum(dim=0) + confusion.sum(dim=1) - confusion.diag()

    if distributed:
        dist.all_reduce(intersection, op=dist.ReduceOp.SUM)
        dist.all_reduce(union, op=dist.ReduceOp.SUM)
    iou = intersection / union.clamp_min(1) * 100.0
    return float(iou.mean().item()), iou.cpu().numpy()


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    model_ema: nn.Module,
    optimizer: AdamW,
    epoch: int,
    previous_best: float,
    previous_best_ema: float,
    best_epoch: int,
    best_epoch_ema: int,
    args: argparse.Namespace,
) -> None:
    torch.save(
        {
            "model": _unwrap(model).state_dict(),
            "model_ema": model_ema.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "previous_best": previous_best,
            "previous_best_ema": previous_best_ema,
            "best_epoch": best_epoch,
            "best_epoch_ema": best_epoch_ema,
            "split": args.split,
            "style_prior_alpha": None if args.no_style_prior else args.alpha,
        },
        path,
    )


def run(args: argparse.Namespace) -> None:
    config_path = args.config.expanduser().resolve()
    cfg = _load_yaml(config_path)
    args.alpha = float(args.alpha if args.alpha is not None else cfg.get("style_prior_alpha", 0.6))
    cfg["epochs"] = int(args.epochs if args.epochs is not None else cfg["epochs"])
    cfg["batch_size"] = int(args.batch_size if args.batch_size is not None else cfg["batch_size"])
    cfg["conf_thresh"] = float(
        args.confidence_threshold if args.confidence_threshold is not None else cfg["conf_thresh"]
    )
    if not 0.0 <= args.alpha:
        raise ValueError("--alpha must be non-negative")
    if not 0.0 <= cfg["conf_thresh"] <= 1.0:
        raise ValueError("--confidence-threshold must be in [0, 1]")
    if cfg["epochs"] <= 0 or cfg["batch_size"] <= 0 or args.workers < 0:
        raise ValueError("epochs and batch size must be positive; workers must be non-negative")

    paths = resolve_paths(args, cfg)
    report = validate_experiment(args, cfg, paths, check_data_files=args.check_only)
    if args.check_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    device, rank, world_size, distributed = _setup_runtime(args)
    _seed_everything(args.seed, rank)
    torch.backends.cudnn.enabled = device.type == "cuda"
    torch.backends.cudnn.benchmark = device.type == "cuda"
    logger = _configure_logger(paths.output_dir, rank, write_file=not args.evaluate_only)
    writer = None
    if rank == 0 and not args.evaluate_only:
        from torch.utils.tensorboard import SummaryWriter

        writer = SummaryWriter(paths.output_dir)
        logger.info("Resolved experiment:\n%s", json.dumps({**report, "world_size": world_size}, indent=2))

    model = _build_model(cfg, device)
    checkpoint_path = _repo_path(args.checkpoint) if args.checkpoint else None
    auto_resume = paths.output_dir / "latest.pth"
    resume_path = None if args.no_resume or args.evaluate_only else (auto_resume if auto_resume.is_file() else None)

    if checkpoint_path:
        payload = _load_torch(checkpoint_path, weights_only=False)
        model.load_state_dict(_select_state(payload, args.checkpoint_key), strict=True)
    elif resume_path is None:
        assert paths.pretrained is not None
        _load_backbone(model, paths.pretrained)

    if args.evaluate_only:
        loader_v = _make_validation_loader(args, cfg, paths, rank, world_size, distributed, device)
        miou, class_iou = evaluate(
            model,
            loader_v,
            device=device,
            num_classes=int(cfg["nclass"]),
            distributed=distributed,
        )
        if rank == 0:
            print(json.dumps({"mIoU": round(miou, 4), "class_IoU": class_iou.tolist()}, indent=2))
        if distributed:
            dist.destroy_process_group()
        return

    loader_l, loader_u, loader_v, sampler_l, sampler_u = _make_loaders(
        args, cfg, paths, rank, world_size, distributed, device
    )

    model_ema = deepcopy(model).eval()
    for parameter in model_ema.parameters():
        parameter.requires_grad_(False)

    if distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DistributedDataParallel(
            model,
            device_ids=[device.index],
            output_device=device.index,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    optimizer = _make_optimizer(_unwrap(model), cfg)
    criterion_l = _make_criterion(cfg, device)
    criterion_u = nn.CrossEntropyLoss(reduction="none").to(device)
    start_epoch = 0
    previous_best = previous_best_ema = 0.0
    best_epoch = best_epoch_ema = -1
    if resume_path:
        payload = _load_torch(resume_path, weights_only=False)
        if payload.get("split") not in (None, args.split):
            raise ValueError(
                f"checkpoint split {payload['split']!r} does not match requested split {args.split!r}"
            )
        _unwrap(model).load_state_dict(_select_state(payload, "model"), strict=True)
        model_ema.load_state_dict(_select_state(payload, "model_ema"), strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        start_epoch = int(payload["epoch"]) + 1
        previous_best = float(payload.get("previous_best", 0.0))
        previous_best_ema = float(payload.get("previous_best_ema", 0.0))
        best_epoch = int(payload.get("best_epoch", -1))
        best_epoch_ema = int(payload.get("best_epoch_ema", -1))
        logger.info("Resumed from %s at epoch %d", resume_path, start_epoch)

    from cpgs.style_prior import inject_style_prior_torch

    total_iterations = len(loader_u) * int(cfg["epochs"])
    completed_steps = 0
    for epoch in range(start_epoch, int(cfg["epochs"])):
        if sampler_l is not None:
            sampler_l.set_epoch(epoch)
        if sampler_u is not None:
            sampler_u.set_epoch(epoch)
        model.train()
        loss_sum = supervised_sum = unsupervised_sum = mask_ratio_sum = 0.0
        log_interval = max(1, len(loader_u) // 8)

        for step, ((image_l, mask_l), unlabeled) in enumerate(zip(loader_l, loader_u)):
            image_w, image_s1, image_s2, ignore_mask, cutmix1, cutmix2, class_prior = unlabeled
            image_l = image_l.to(device, non_blocking=True)
            mask_l = mask_l.to(device, non_blocking=True)
            image_w = image_w.to(device, non_blocking=True)
            image_s1 = image_s1.to(device, non_blocking=True)
            image_s2 = image_s2.to(device, non_blocking=True)
            ignore_mask = ignore_mask.to(device, non_blocking=True)
            cutmix1 = cutmix1.to(device, non_blocking=True)
            cutmix2 = cutmix2.to(device, non_blocking=True)

            with torch.no_grad():
                teacher_logits = model_ema(image_w)
                if not args.no_style_prior:
                    teacher_logits = inject_style_prior_torch(teacher_logits, class_prior, alpha=args.alpha)
                teacher_probability = teacher_logits.softmax(dim=1)
                confidence, pseudo_mask = teacher_probability.max(dim=1)

            box1 = cutmix1.unsqueeze(1).expand_as(image_s1).bool()
            box2 = cutmix2.unsqueeze(1).expand_as(image_s2).bool()
            image_s1[box1] = image_s1.flip(0)[box1]
            image_s2[box2] = image_s2.flip(0)[box2]

            logits_l = model(image_l)
            logits_s1, logits_s2 = model(torch.cat((image_s1, image_s2)), comp_drop=True).chunk(2)

            pseudo1, confidence1, ignore1 = pseudo_mask.clone(), confidence.clone(), ignore_mask.clone()
            pseudo2, confidence2, ignore2 = pseudo_mask.clone(), confidence.clone(), ignore_mask.clone()
            pseudo1[cutmix1 == 1] = pseudo_mask.flip(0)[cutmix1 == 1]
            confidence1[cutmix1 == 1] = confidence.flip(0)[cutmix1 == 1]
            ignore1[cutmix1 == 1] = ignore_mask.flip(0)[cutmix1 == 1]
            pseudo2[cutmix2 == 1] = pseudo_mask.flip(0)[cutmix2 == 1]
            confidence2[cutmix2 == 1] = confidence.flip(0)[cutmix2 == 1]
            ignore2[cutmix2 == 1] = ignore_mask.flip(0)[cutmix2 == 1]

            loss_l = criterion_l(logits_l, mask_l)
            valid1 = ignore1 != 255
            valid2 = ignore2 != 255
            accepted1 = valid1 & (confidence1 >= float(cfg["conf_thresh"]))
            accepted2 = valid2 & (confidence2 >= float(cfg["conf_thresh"]))
            loss_s1 = (criterion_u(logits_s1, pseudo1) * accepted1).sum() / valid1.sum().clamp_min(1)
            loss_s2 = (criterion_u(logits_s2, pseudo2) * accepted2).sum() / valid2.sum().clamp_min(1)
            loss_u = (loss_s1 + loss_s2) / 2.0
            loss = (loss_l + loss_u) / 2.0

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            iteration = epoch * len(loader_u) + step
            learning_rate = float(cfg["lr"]) * (1 - iteration / total_iterations) ** 0.9
            optimizer.param_groups[0]["lr"] = learning_rate
            optimizer.param_groups[1]["lr"] = learning_rate * float(cfg["lr_multi"])
            ema_ratio = min(1 - 1 / (iteration + 1), 0.996)
            with torch.no_grad():
                for current, average in zip(_unwrap(model).parameters(), model_ema.parameters()):
                    average.mul_(ema_ratio).add_(current.detach(), alpha=1 - ema_ratio)
                for current, average in zip(_unwrap(model).buffers(), model_ema.buffers()):
                    average.copy_(current)

            valid_weak = ignore_mask != 255
            accepted_weak = (confidence >= float(cfg["conf_thresh"])) & valid_weak
            mask_ratio = float(accepted_weak.sum() / valid_weak.sum().clamp_min(1))
            loss_sum += float(loss.detach())
            supervised_sum += float(loss_l.detach())
            unsupervised_sum += float(loss_u.detach())
            mask_ratio_sum += mask_ratio
            completed_steps += 1
            count = step + 1
            if rank == 0 and writer is not None:
                writer.add_scalar("train/loss_all", float(loss.detach()), iteration)
                writer.add_scalar("train/loss_x", float(loss_l.detach()), iteration)
                writer.add_scalar("train/loss_s", float(loss_u.detach()), iteration)
                writer.add_scalar("train/mask_ratio", mask_ratio, iteration)
            if rank == 0 and step % log_interval == 0:
                logger.info(
                    "Epoch %d step %d/%d: lr=%.7f loss=%.3f supervised=%.3f unsupervised=%.3f mask=%.3f",
                    epoch,
                    step,
                    len(loader_u),
                    learning_rate,
                    loss_sum / count,
                    supervised_sum / count,
                    unsupervised_sum / count,
                    mask_ratio_sum / count,
                )
            if args.max_steps is not None and completed_steps >= args.max_steps:
                break

        performed_validation = not args.skip_validation and not (
            args.max_steps is not None and completed_steps >= args.max_steps
        )
        if not performed_validation:
            miou = miou_ema = 0.0
            class_iou = class_iou_ema = np.zeros(int(cfg["nclass"]), dtype=np.float64)
        else:
            miou, class_iou = evaluate(
                _unwrap(model), loader_v, device=device, num_classes=int(cfg["nclass"]), distributed=distributed
            )
            miou_ema, class_iou_ema = evaluate(
                model_ema, loader_v, device=device, num_classes=int(cfg["nclass"]), distributed=distributed
            )

        if rank == 0:
            if performed_validation:
                for class_index, (score, score_ema) in enumerate(zip(class_iou, class_iou_ema)):
                    logger.info("Class %02d IoU: %.2f, EMA: %.2f", class_index, score, score_ema)
                logger.info("Epoch %d mIoU: %.2f, EMA: %.2f", epoch, miou, miou_ema)
            else:
                logger.info("Epoch %d validation skipped by smoke-test options", epoch)
            is_best_ema = performed_validation and miou_ema >= previous_best_ema
            previous_best = max(miou, previous_best)
            previous_best_ema = max(miou_ema, previous_best_ema)
            if miou >= previous_best:
                best_epoch = epoch
            if miou_ema >= previous_best_ema:
                best_epoch_ema = epoch
            _save_checkpoint(
                paths.output_dir / "latest.pth",
                model,
                model_ema,
                optimizer,
                epoch,
                previous_best,
                previous_best_ema,
                best_epoch,
                best_epoch_ema,
                args,
            )
            if is_best_ema:
                _save_checkpoint(
                    paths.output_dir / "best.pth",
                    model,
                    model_ema,
                    optimizer,
                    epoch,
                    previous_best,
                    previous_best_ema,
                    best_epoch,
                    best_epoch_ema,
                    args,
                )
            if writer is not None:
                writer.add_scalar("eval/mIoU", miou, epoch)
                writer.add_scalar("eval/mIoU_ema", miou_ema, epoch)

        if distributed:
            dist.barrier()
        if args.max_steps is not None and completed_steps >= args.max_steps:
            break

    if writer is not None:
        writer.close()
    if distributed:
        dist.destroy_process_group()


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()
