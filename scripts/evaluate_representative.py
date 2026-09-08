"""Evaluate the representative 1/2 C-PGS checkpoint on the ACA validation set.

The command intentionally accepts no parameters, as required by GRSI. External
assets are expected at the repository-relative locations documented in README.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml

from cpgs.experiment_ip import refine_with_experiment_instance_prior


EXPECTED_CHECKPOINT_SHA256 = (
    "347db73172344905182250feef1dddd5bd506bb516b8fdc716274425a88741fc"
)
EXPECTED_RESULTS = {"SP": 64.48, "SP+IP": 65.16}
RESULT_TOLERANCE = 0.05


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_split(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) != 2:
            raise ValueError(f"{path}:{line_number}: expected image and label paths")
        pairs.append((fields[0], fields[1]))
    if not pairs:
        raise ValueError(f"validation split is empty: {path}")
    return pairs


def _resolve_instance_path(label_path: Path) -> Path:
    parts = list(label_path.parts)
    try:
        directory_index = parts.index("SegmentationClassNpy")
    except ValueError as exc:
        raise ValueError(
            f"semantic label path does not contain SegmentationClassNpy: {label_path}"
        ) from exc
    parts[directory_index] = "InstanceClassNpy"
    return Path(*parts)


def _load_label(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.asarray(np.load(path), dtype=np.int64)
    with Image.open(path) as image:
        return np.asarray(image, dtype=np.int64)


def _update_confusion(
    matrix: np.ndarray,
    prediction: np.ndarray,
    target: np.ndarray,
    *,
    num_classes: int,
    ignore_index: int,
) -> None:
    if prediction.shape != target.shape:
        raise ValueError(
            f"prediction shape {prediction.shape} differs from target {target.shape}"
        )
    valid = (target != ignore_index) & (target >= 0) & (target < num_classes)
    values = target[valid] * num_classes + prediction[valid]
    matrix += np.bincount(
        values, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def _miou_percent(matrix: np.ndarray) -> tuple[float, np.ndarray]:
    matrix_float = matrix.astype(np.float64)
    intersection = np.diag(matrix_float)
    union = matrix_float.sum(axis=1) + matrix_float.sum(axis=0) - intersection
    per_class = intersection / (union + 1e-10) * 100.0
    return float(np.mean(per_class)), per_class


def _build_model(repository: Path, checkpoint_path: Path, device):
    integration = repository / "experiments" / "integrations" / "unimatch_v2"
    sys.path.insert(0, str(integration))
    from model.semseg.dpt import DPT

    model = DPT(
        encoder_size="small",
        nclass=18,
        features=64,
        out_channels=[48, 96, 192, 384],
        device=device,
    )

    import torch

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if "model_ema" not in checkpoint:
        raise KeyError("checkpoint does not contain the required model_ema state")
    state = checkpoint["model_ema"]
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def _image_tensor(path: Path):
    import torch

    with Image.open(path) as image:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def _predict(model, image_path: Path, device) -> np.ndarray:
    import torch
    import torch.nn.functional as functional

    image = _image_tensor(image_path).to(device)
    original_height, original_width = image.shape[-2:]
    resized_height = max(14, int(original_height / 14 + 0.5) * 14)
    resized_width = max(14, int(original_width / 14 + 0.5) * 14)
    image = functional.interpolate(
        image,
        (resized_height, resized_width),
        mode="bilinear",
        align_corners=True,
    )
    with torch.inference_mode():
        logits = model(image)
        logits = functional.interpolate(
            logits,
            (original_height, original_width),
            mode="bilinear",
            align_corners=True,
        )
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int64)


def _write_results(
    output_directory: Path,
    results: list[tuple[str, float, np.ndarray]],
    class_names: list[str],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    csv_path = output_directory / "representative_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "mean_iou_percent", *class_names])
        for mode, mean_iou, per_class in results:
            writer.writerow(
                [mode, f"{mean_iou:.6f}", *(f"{value:.6f}" for value in per_class)]
            )

    markdown_path = output_directory / "representative_results.md"
    lines = [
        "# Representative ACA evaluation",
        "",
        "| Mode | Reproduced mIoU (%) | Paper mIoU (%) | Difference |",
        "|---|---:|---:|---:|",
    ]
    for mode, mean_iou, _ in results:
        expected = EXPECTED_RESULTS[mode]
        lines.append(
            f"| {mode} | {mean_iou:.2f} | {expected:.2f} | {mean_iou - expected:+.2f} |"
        )
    lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    import torch

    repository = Path(__file__).resolve().parents[1]
    config_path = repository / "configs" / "aca.yaml"
    split_path = repository / "data" / "splits" / "aca" / "val.txt"
    dataset_root = repository / "data" / "ACA"
    checkpoint_path = (
        repository / "checkpoints" / "unimatch_v2_sp_aca_1_2_best.pth"
    )

    missing = [
        path
        for path in (config_path, split_path, dataset_root, checkpoint_path)
        if not path.exists()
    ]
    if missing:
        print("Representative evaluation assets are missing:", file=sys.stderr)
        for path in missing:
            print(f"- {path}", file=sys.stderr)
        print(
            "The ACA data and checkpoint are distributed separately; place them at "
            "the repository-relative paths documented in README.md.",
            file=sys.stderr,
        )
        return 2

    actual_hash = _sha256(checkpoint_path)
    if actual_hash != EXPECTED_CHECKPOINT_SHA256:
        print(
            "Checkpoint SHA-256 mismatch:\n"
            f"expected: {EXPECTED_CHECKPOINT_SHA256}\nactual:   {actual_hash}",
            file=sys.stderr,
        )
        return 2

    configuration = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    dataset_config = configuration["dataset"]
    instance_config = configuration["instance_prior"]
    num_classes = int(dataset_config["num_classes"])
    ignore_index = int(dataset_config["ignore_index"])
    class_names = list(dataset_config["classes"])
    pairs = _read_split(split_path)
    if len(pairs) != int(dataset_config["validation_images"]):
        raise ValueError(
            f"expected {dataset_config['validation_images']} validation images; "
            f"found {len(pairs)}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("WARNING: CUDA is unavailable; evaluation on CPU will be slow.")
    print(f"Device: {device}")
    print(f"Validation images: {len(pairs)}")
    model = _build_model(repository, checkpoint_path, device)

    sp_confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    sp_ip_confusion = np.zeros_like(sp_confusion)
    for index, (image_relative, label_relative) in enumerate(pairs, start=1):
        image_path = dataset_root / Path(image_relative)
        label_path = dataset_root / Path(label_relative)
        instance_path = _resolve_instance_path(label_path)
        for required in (image_path, label_path, instance_path):
            if not required.is_file():
                raise FileNotFoundError(f"missing ACA validation asset: {required}")

        target = _load_label(label_path)
        instance_ids = _load_label(instance_path)
        prediction = _predict(model, image_path, device)
        refined = refine_with_experiment_instance_prior(
            prediction,
            instance_ids,
            num_classes=num_classes,
            dominance_threshold=float(instance_config["dominance_threshold"]),
            normalized_expansion=float(instance_config["normalized_expansion"]),
            radius_scale=float(instance_config["radius_scale"]),
        )
        _update_confusion(
            sp_confusion,
            prediction,
            target,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        _update_confusion(
            sp_ip_confusion,
            refined,
            target,
            num_classes=num_classes,
            ignore_index=ignore_index,
        )
        print(f"[{index:03d}/{len(pairs):03d}] {image_relative}")

    sp_mean, sp_per_class = _miou_percent(sp_confusion)
    sp_ip_mean, sp_ip_per_class = _miou_percent(sp_ip_confusion)
    results = [
        ("SP", sp_mean, sp_per_class),
        ("SP+IP", sp_ip_mean, sp_ip_per_class),
    ]
    _write_results(repository / "outputs", results, class_names)

    failed = False
    for mode, value, _ in results:
        expected = EXPECTED_RESULTS[mode]
        difference = value - expected
        print(
            f"{mode}: {value:.2f} mIoU "
            f"(paper {expected:.2f}, difference {difference:+.2f})"
        )
        if abs(difference) > RESULT_TOLERANCE:
            failed = True
    if failed:
        print(
            f"RESULT CHECK: FAIL (tolerance ±{RESULT_TOLERANCE:.2f})",
            file=sys.stderr,
        )
        return 1
    print(f"RESULT CHECK: PASS (tolerance ±{RESULT_TOLERANCE:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
