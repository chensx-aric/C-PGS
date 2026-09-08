"""Check that the public C-PGS folder is safe and structurally complete."""

from __future__ import annotations

from pathlib import Path


REQUIRED_FILES = {
    ".gitignore",
    "CITATION.cff",
    "LICENSE",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "assets/cpgs_framework.png",
    "assets/cpgs_representative.png",
    "configs/aca.yaml",
    "configs/assets.example.yaml",
    "data/README.md",
    "docs/GRSI_REPRODUCTION.md",
    "docs/TRAINING.md",
    "evaluate_representative.py",
    "experiments/baseline/unimatch_v2_official/LICENSE",
    "experiments/integrations/unimatch_v2/LICENSE",
    "experiments/integrations/unimatch_v2/MODIFICATIONS.md",
    "experiments/integrations/unimatch_v2/train.py",
    "install.sh",
    "requirements-evaluation.txt",
    "requirements-full.txt",
    "scripts/evaluate_representative.py",
    "src/cpgs/style_prior.py",
    "src/cpgs/instance_prior.py",
}

EXPECTED_SPLIT_COUNTS = {
    "train_full.txt": 1308,
    "train_labeled_0.0625.txt": 82,
    "train_labeled_0.125.txt": 164,
    "train_labeled_0.25.txt": 327,
    "train_labeled_0.5.txt": 654,
    "train_unlabeled_0.0625.txt": 1226,
    "train_unlabeled_0.125.txt": 1144,
    "train_unlabeled_0.25.txt": 981,
    "train_unlabeled_0.5.txt": 654,
    "val.txt": 145,
}

FORBIDDEN_SUFFIXES = {".ckpt", ".log", ".npy", ".pt", ".pth", ".tar", ".zip"}
FORBIDDEN_DIRECTORIES = {
    ".pytest_cache",
    ".venv",
    "__pycache__",
    "checkpoints",
    "precomputed",
}
PRIVATE_MARKERS = (
    "/public/home/",
    "chensixin_pfr",
    "C:\\Users\\",
    "D:\\",
)
AUTHOR_EMAILS = (
    "csx2494885979@163.com",
    "zhangsl@tyust.edu.cn",
    "2024008@tyust.edu.cn",
    "bubblezhou1994@163.com",
    "pengfr@sxu.edu.cn",
    "hlh@tyust.edu.cn",
)


def main() -> int:
    repository = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for relative in sorted(REQUIRED_FILES):
        if not (repository / relative).is_file():
            failures.append(f"missing required file: {relative}")

    integration = repository / "experiments" / "integrations" / "unimatch_v2"
    entry_points = sorted(path.name for path in integration.glob("*.py"))
    if entry_points != ["train.py"]:
        failures.append(
            "UniMatch-V2 integration must have exactly one top-level Python entry "
            f"point (train.py); found {entry_points}"
        )

    split_root = repository / "data" / "splits" / "aca"
    for name, expected_count in EXPECTED_SPLIT_COUNTS.items():
        path = split_root / name
        if not path.is_file():
            failures.append(f"missing split manifest: {name}")
            continue
        lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
        if len(lines) != expected_count:
            failures.append(
                f"{name}: expected {expected_count} rows, found {len(lines)}"
            )
        if any("\\" in line or (len(line) > 1 and line[1] == ":") for line in lines):
            failures.append(f"{name}: contains a non-portable path")

    for path in repository.rglob("*"):
        if ".git" in path.relative_to(repository).parts:
            continue
        relative = path.relative_to(repository).as_posix()
        if path.is_dir() and path.name in FORBIDDEN_DIRECTORIES:
            failures.append(f"forbidden generated/private directory: {relative}")
        if not path.is_file():
            continue
        if path.suffix.lower() in FORBIDDEN_SUFFIXES or path.name.endswith(".tar.gz"):
            failures.append(f"private/generated asset in public folder: {relative}")
        if path.stat().st_size > 50 * 1024 * 1024:
            failures.append(f"file exceeds 50 MiB: {relative}")
        if path.resolve() == Path(__file__).resolve():
            continue
        if path.suffix.lower() not in {".md", ".py", ".sh", ".ps1", ".txt", ".toml", ".yaml", ".yml", ".cff", ".csv"}:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for marker in PRIVATE_MARKERS:
            if marker in text:
                failures.append(f"private path marker in {relative}: {marker}")
        for email in AUTHOR_EMAILS:
            if email in text:
                failures.append(f"author email in public file: {relative}")

    if (repository / "data" / "ACA").exists():
        failures.append("ACA dataset must not be in the public folder")
    if any("allspark" in path.name.lower() for path in repository.rglob("*")):
        failures.append("AllSpark content must not be included")

    if failures:
        print("PUBLIC RELEASE: FAIL")
        for failure in sorted(set(failures)):
            print(f"- {failure}")
        return 1

    print("PUBLIC RELEASE: PASS")
    print("One UniMatch-V2 entry point and all expected ACA split counts were verified.")
    print("No dataset arrays, weights, caches, logs, author emails, or private paths found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
