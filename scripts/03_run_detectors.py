"""
Stage 03: Run detector inference with released pretrained checkpoints.

Current thesis default detector set:
- cnndetection
- lgrad
- npr

This script still keeps optional compatibility backends (for example f3net,
gram, univfd, dire), but they are not part of the current default experiment
path unless explicitly requested via --detector.

Notes
-----
1) The script is designed for Colab/runtime execution. It downloads source zips
   and weights on demand into `.external_models/`.
2) CNNDetection uses the official Wang et al. detector family; LGrad uses the
   public LGrad checkpoints; NPR uses the official NPR release.
3) All rows are marked as split='test' because these are pretrained models,
   not train/test-split logistic baselines.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import os
import site
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm


SIDBENCH_ARCHIVE_URL = "https://codeload.github.com/mever-team/sidbench/zip/refs/heads/main"
SIDBENCH_WEIGHTS_URL = "https://drive.google.com/file/d/1YuJ2so_1LgOSRjJUqZL-L2EQmuJcdxQh/view?usp=sharing"
PYDEEPFAKEDET_ARCHIVE_URL = "https://codeload.github.com/wdrink/PyDeepFakeDet/zip/refs/heads/main"
LGRAD_OFFICIAL_ARCHIVE_URL = "https://codeload.github.com/chuangchuangtan/LGrad/zip/refs/heads/master"
CNNDETECTION_OFFICIAL_ARCHIVE_URL = "https://codeload.github.com/PeterWang512/CNNDetection/zip/refs/heads/master"
NPR_OFFICIAL_ARCHIVE_URL = "https://codeload.github.com/chuangchuangtan/NPR-DeepfakeDetection/zip/refs/heads/main"
F3NET_RAW_CKPT_URL = "https://drive.google.com/file/d/1mUNeR-r5vi-dNtxw4wIBxGaLZculxfQR/view?usp=sharing"


@dataclass(frozen=True)
class DetectorConfig:
    name: str
    backend: str
    display_name: str
    sidbench_model_name: str | None = None
    sidbench_ckpt_relpath: str | None = None
    sidbench_extra: Tuple[str, ...] = ()


DETECTOR_CONFIGS: Dict[str, DetectorConfig] = {
    "cnndetection": DetectorConfig(
        name="cnndetection",
        backend="cnndetection_official",
        display_name="CNNDetection",
        sidbench_ckpt_relpath="weights/cnndetect/blur_jpg_prob0.5.pth",
    ),
    "gram": DetectorConfig(
        name="gram",
        backend="sidbench",
        display_name="GramNet",
        sidbench_model_name="GramNet",
        sidbench_ckpt_relpath="weights/gramnet/Gram.pth",
        sidbench_extra=("--resizeSize", "299", "--cropSize", "299"),
    ),
    "univfd": DetectorConfig(
        name="univfd",
        backend="sidbench",
        display_name="UnivFD",
        sidbench_model_name="UnivFD",
        sidbench_ckpt_relpath="weights/univfd/fc_weights.pth",
        sidbench_extra=("--resizeSize", "224", "--cropSize", "224"),
    ),
    "dire": DetectorConfig(
        name="dire",
        backend="sidbench",
        display_name="DIRE",
        sidbench_model_name="Dire",
        sidbench_ckpt_relpath="weights/dire/lsun_adm.pth",
        sidbench_extra=(
            "--resizeSize",
            "256",
            "--cropSize",
            "256",
            "--DireGenerativeModelPath",
            "weights/preprocessing/lsun_bedroom.pt",
        ),
    ),
    "lgrad": DetectorConfig(
        name="lgrad",
        backend="lgrad_official",
        display_name="LGrad",
        sidbench_ckpt_relpath="weights/lgrad/LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
    ),
    "npr": DetectorConfig(
        name="npr",
        backend="npr_official",
        display_name="NPR",
        sidbench_ckpt_relpath="weights/npr/NPR.pth",
    ),
    "f3net": DetectorConfig(
        name="f3net",
        backend="pydeepfakedet_f3net",
        display_name="F3Net",
    ),
}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--metadata-in", type=Path, default=root / "data" / "metadata_balanced.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "detector_outputs")
    parser.add_argument(
        "--detector",
        type=str,
        default="cnndetection",
        help=(
            "Detector name, comma-separated list, or 'all'. "
            "Current thesis default set is cnndetection,lgrad,npr."
        ),
    )
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="If set, scan this directory for images instead of using --metadata-in CSV.")
    parser.add_argument("--input-csv", type=Path, default=None,
                        help="Alias for --metadata-in for pipeline compatibility.")
    parser.add_argument("--external-root", type=Path, default=root / ".external_models")
    parser.add_argument("--keep-staging", action="store_true")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference device for in-process detector backends. Default is cuda.",
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for in-process detector inference.")
    parser.add_argument("--num-workers", type=int, default=2, help="Dataloader workers for in-process backends.")
    parser.add_argument("--pin-memory", action="store_true", help="Enable DataLoader pin_memory on CUDA.")
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable automatic mixed precision for compatible official detector backends on CUDA.",
    )
    return parser.parse_args()


def parse_detector_names(detector_arg: str) -> List[str]:
    token = detector_arg.strip().lower()
    if token == "all":
        return list(DETECTOR_CONFIGS.keys())

    names = [part.strip().lower() for part in detector_arg.split(",") if part.strip()]
    if not names:
        raise ValueError("No detector names were provided.")

    invalid = [name for name in names if name not in DETECTOR_CONFIGS]
    if invalid:
        raise ValueError(
            f"Unsupported detector(s): {', '.join(invalid)}. "
            f"Valid choices: {', '.join(sorted(DETECTOR_CONFIGS))}, all"
        )
    return names


def validate_runtime(device: str) -> None:
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was requested for detector inference, but no GPU is available. "
                "In Colab, switch Runtime -> Change runtime type -> GPU, "
                "or pass --device cpu if you intentionally want CPU mode."
            )


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class ImagePathDataset(Dataset):
    def __init__(self, paths: List[str], transform, crop_even: bool = False):
        self.paths = paths
        self.transform = transform
        self.crop_even = crop_even

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.crop_even:
            width, height = img.size
            if width % 2 == 1 or height % 2 == 1:
                img = img.crop((0, 0, width - (width % 2), height - (height % 2)))
        tensor = self.transform(img)
        return path, tensor


def build_loader(paths: List[str], transform, args: argparse.Namespace, crop_even: bool = False) -> DataLoader:
    dataset = ImagePathDataset(paths, transform=transform, crop_even=crop_even)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory and args.device == "cuda"),
    )


def amp_enabled(args: argparse.Namespace) -> bool:
    return bool(args.amp and args.device == "cuda")


def run(cmd: List[str], cwd: Path | None = None) -> None:
    print("[cmd]", " ".join(cmd))
    subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True)


def ensure_python_pkg(pkg_name: str) -> None:
    try:
        importlib.import_module(pkg_name)
    except ImportError:
        run([sys.executable, "-m", "pip", "install", pkg_name])


def ensure_requirements(requirements_file: Path, marker: Path) -> None:
    if marker.exists():
        return
    if requirements_file.exists():
        run([sys.executable, "-m", "pip", "install", "-r", str(requirements_file)])
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("ok", encoding="utf-8")


def resolve_existing_path(candidates: Iterable[Path]) -> Path | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_sidbench_checkpoint(args: argparse.Namespace, cfg: DetectorConfig) -> Path:
    external_root = args.external_root
    candidates: List[Path] = []

    if cfg.name == "cnndetection":
        candidates.extend(
            [
                external_root / "weights" / "cnndetect" / "blur_jpg_prob0.5.pth",
                external_root / "sidbench_weights" / "weights" / "cnndetect" / "blur_jpg_prob0.5.pth",
            ]
        )
    elif cfg.name == "gram":
        candidates.extend(
            [
                external_root / "weights" / "gram" / "gram_resnet50.pth",
                external_root / "weights" / "gramnet" / "Gram.pth",
                external_root / "sidbench_weights" / "weights" / "gram" / "gram_resnet50.pth",
                external_root / "sidbench_weights" / "weights" / "gramnet" / "Gram.pth",
            ]
        )
    elif cfg.name == "univfd":
        candidates.extend(
            [
                external_root / "weights" / "univfd" / "fc_weights.pth",
                external_root / "weights" / "univfd" / "fc_weights..pth",
                external_root / "sidbench_weights" / "weights" / "univfd" / "fc_weights.pth",
                external_root / "sidbench_weights" / "weights" / "univfd" / "fc_weights..pth",
            ]
        )
    elif cfg.name == "dire":
        candidates.extend(
            [
                external_root / "weights" / "dire" / "lsun_adm.pth",
                external_root / "weights" / "dire" / "lsun_iddpm.pth",
                external_root / "weights" / "dire" / "lsun_pndm.pth",
                external_root / "weights" / "dire" / "lsun_stylegan.pth",
                external_root / "sidbench_weights" / "weights" / "dire" / "lsun_adm.pth",
                external_root / "sidbench_weights" / "weights" / "dire" / "lsun_iddpm.pth",
                external_root / "sidbench_weights" / "weights" / "dire" / "lsun_pndm.pth",
                external_root / "sidbench_weights" / "weights" / "dire" / "lsun_stylegan.pth",
            ]
        )
    resolved = resolve_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            f"Missing local checkpoint for {cfg.name}. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )
    return resolved


def resolve_lgrad_checkpoint(args: argparse.Namespace) -> Path:
    candidates = [
        args.external_root / "weights" / "lgrad" / "Lgrad_Mix.pth",
        args.external_root / "weights" / "lgrad" / "LGrad-4class-Trainon-Progan_car_cat_chair_horse.pth",
        args.external_root / "weights" / "lgrad" / "LGrad-2class-Trainon-Progan_chair_horse.pth",
        args.external_root / "weights" / "lgrad" / "LGrad-1class-Trainon-Progan_horse.pth",
    ]
    resolved = resolve_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Missing local LGrad checkpoint. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )
    return resolved


def resolve_cnndetection_checkpoint(args: argparse.Namespace) -> Path:
    candidates = [
        args.external_root / "weights" / "cnndetect" / "blur_jpg_prob0.5.pth",
        args.external_root / "weights" / "cnndetection" / "blur_jpg_prob0.5.pth",
        args.external_root / "sidbench_weights" / "weights" / "cnndetect" / "blur_jpg_prob0.5.pth",
    ]
    resolved = resolve_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Missing local CNNDetection checkpoint. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )
    return resolved


def resolve_lgrad_preprocessing_ckpt(args: argparse.Namespace) -> Path:
    candidates = [
        args.external_root / "weights" / "preprocessing" / "karras2019stylegan-bedrooms-256x256_discriminator.pth",
    ]
    resolved = resolve_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Missing LGrad preprocessing checkpoint. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )
    return resolved


def resolve_npr_checkpoint(args: argparse.Namespace) -> Path:
    candidates = [
        args.external_root / "weights" / "npr" / "NPR.pth",
        args.external_root / "weights" / "npr" / "model_epoch_last_3090.pth",
        args.external_root / "NPR-DeepfakeDetection" / "NPR.pth",
        args.external_root / "NPR-DeepfakeDetection" / "model_epoch_last_3090.pth",
    ]
    resolved = resolve_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Missing local NPR checkpoint. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )
    return resolved


def resolve_dire_preprocessing_ckpt(args: argparse.Namespace) -> Path:
    candidates = [
        args.external_root / "weights" / "preprocessing" / "lsun_bedroom.pt",
        args.external_root / "sidbench_weights" / "weights" / "preprocessing" / "lsun_bedroom.pt",
    ]
    resolved = resolve_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Missing DIRE preprocessing checkpoint. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )
    return resolved


def resolve_f3net_checkpoint(args: argparse.Namespace) -> Path:
    candidates = [
        args.external_root / "weights" / "f3net" / "F3Net_Mix.pth",
        args.external_root / "weights" / "f3net_raw.pth",
        args.external_root / "weights" / "F3Net_Mix.pth",
    ]
    resolved = resolve_existing_path(candidates)
    if resolved is None:
        raise FileNotFoundError(
            "Missing local F3Net checkpoint. Checked:\n"
            + "\n".join(str(path) for path in candidates)
        )
    return resolved


def download_file(url: str, destination: Path) -> Path:
    ensure_dir(destination.parent)
    if destination.exists():
        return destination
    print(f"[download] {url} -> {destination}")
    urlretrieve(url, destination)
    return destination


def download_gdrive(url: str, destination: Path) -> Path:
    ensure_python_pkg("gdown")
    ensure_dir(destination.parent)
    if destination.exists():
        return destination
    run([sys.executable, "-m", "gdown", "--fuzzy", url, "-O", str(destination)])
    return destination


def ensure_archive(url: str, destination_dir: Path, folder_hint: str) -> Path:
    marker = destination_dir / ".ready"
    if marker.exists():
        return destination_dir
    ensure_dir(destination_dir.parent)
    zip_path = destination_dir.parent / f"{folder_hint}.zip"
    download_file(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(destination_dir.parent)
    extracted_candidates = [p for p in destination_dir.parent.iterdir() if p.is_dir() and p.name.startswith(folder_hint)]
    if not extracted_candidates:
        raise FileNotFoundError(f"Could not find extracted folder for {folder_hint} in {destination_dir.parent}")
    extracted_root = extracted_candidates[0]
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    extracted_root.rename(destination_dir)
    marker.write_text("ok", encoding="utf-8")
    return destination_dir


def patch_sidbench_transformers_compat(repo_root: Path) -> None:
    target = repo_root / "networks" / "med.py"
    if not target.exists():
        return

    text = target.read_text(encoding="utf-8")
    old = "from transformers.modeling_utils import apply_chunking_to_forward"
    new = (
        "try:\n"
        "    from transformers.modeling_utils import apply_chunking_to_forward\n"
        "except ImportError:\n"
        "    from transformers.pytorch_utils import apply_chunking_to_forward"
    )
    if old in text and new not in text:
        text = text.replace(old, new)
        target.write_text(text, encoding="utf-8")
        print(f"[patch] updated transformers compatibility in {target}")


def build_df_from_dir(input_dir: Path) -> pd.DataFrame:
    rows = []
    for label_dir in sorted(input_dir.iterdir()):
        if not label_dir.is_dir():
            continue
        label_name = label_dir.name
        y_true = 1 if label_name == "fake" else 0
        for group_dir in sorted(label_dir.iterdir()):
            if not group_dir.is_dir():
                continue
            group = group_dir.name
            for img in sorted(group_dir.glob("*.png")) + sorted(group_dir.glob("*.jpg")):
                rows.append({
                    "id": img.stem,
                    "group": group,
                    "y_true": y_true,
                    "file_path": str(img),
                    "prompt": "",
                })
    return pd.DataFrame(rows)


def load_input_dataframe(args: argparse.Namespace) -> pd.DataFrame:
    if args.input_csv is not None:
        args.metadata_in = args.input_csv
    if args.input_dir is not None:
        df = build_df_from_dir(args.input_dir)
        print(f"[info] Built metadata from directory: {args.input_dir} ({len(df)} images)")
    else:
        df = pd.read_csv(args.metadata_in)
    if len(df) == 0:
        raise ValueError(f"Empty metadata/input for detector={args.detector}")
    return df.reset_index(drop=True)


def safe_link_or_copy(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def stage_dataset(df: pd.DataFrame, staging_root: Path) -> Tuple[Path, Dict[str, Dict[str, object]]]:
    if staging_root.exists():
        shutil.rmtree(staging_root)
    path_map: Dict[str, Dict[str, object]] = {}
    for idx, row in df.iterrows():
        src = Path(str(row["file_path"])).resolve()
        group = str(row["group"])
        label_dir = "1_fake" if int(row["y_true"]) == 1 else "0_real"
        ext = src.suffix.lower() or ".png"
        staged_name = f"{idx:06d}_{src.stem}{ext}"
        dst = staging_root / group / label_dir / staged_name
        safe_link_or_copy(src, dst)
        path_map[str(dst.resolve())] = {
            "index": idx,
            "group": group,
            "y_true": int(row["y_true"]),
            "src_path": str(src),
        }
    return staging_root, path_map


def resolve_prediction_path(path_str: str) -> str:
    return str(Path(path_str).resolve())


def save_outputs(df: pd.DataFrame, args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_csv = args.output_dir / f"{args.detector}_scores.csv"
    summary_path = args.output_dir / f"{args.detector}_summary.txt"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    y = df["y_true"].astype(int).to_numpy()
    score = df["score"].astype(float).to_numpy()
    auc = float("nan")
    if len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, score))
    acc = float(np.mean(y == df["y_hat"].astype(int).to_numpy()))
    summary_lines = [
        f"detector={args.detector}",
        "backend=pretrained",
        f"accuracy={acc:.6f}",
        f"auc={auc:.6f}",
        f"n_total={len(df)}",
    ]
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"[saved] detector outputs: {output_csv}")
    print(f"[saved] summary: {summary_path}")


def run_sidbench(df: pd.DataFrame, args: argparse.Namespace, cfg: DetectorConfig) -> pd.DataFrame:
    repo_root = ensure_archive(SIDBENCH_ARCHIVE_URL, args.external_root / "sidbench", "sidbench-main")
    patch_sidbench_transformers_compat(repo_root)
    ensure_requirements(repo_root / "requirements.txt", args.external_root / ".installed" / "sidbench.ok")

    staging_root, path_map = stage_dataset(df, args.external_root / "staging" / cfg.name)
    predictions_file = args.external_root / "predictions" / f"{cfg.name}_sidbench.csv"
    ensure_dir(predictions_file.parent)

    checkpoint = resolve_sidbench_checkpoint(args, cfg)

    cmd = [
        sys.executable,
        "test.py",
        "--dataPath",
        str(staging_root),
        "--modelName",
        str(cfg.sidbench_model_name),
        "--ckpt",
        str(checkpoint),
        "--predictionsFile",
        str(predictions_file),
    ]

    extra_args = list(cfg.sidbench_extra)
    if cfg.name == "lgrad":
        for i, token in enumerate(extra_args):
            if token == "weights/preprocessing/karras2019stylegan-bedrooms-256x256_discriminator.pth":
                extra_args[i] = str(resolve_lgrad_preprocessing_ckpt(args))
    elif cfg.name == "dire":
        for i, token in enumerate(extra_args):
            if token == "weights/preprocessing/lsun_bedroom.pt":
                extra_args[i] = str(resolve_dire_preprocessing_ckpt(args))
    cmd.extend(extra_args)
    run(cmd, cwd=repo_root)

    pred_df = pd.read_csv(predictions_file)
    out = df.copy()
    out["detector_name"] = cfg.name
    out["split"] = "test"
    out["score"] = np.nan
    out["y_hat"] = -1

    for _, row in pred_df.iterrows():
        pred_path = resolve_prediction_path(str(row["Image Path"]))
        meta = path_map.get(pred_path)
        if meta is None:
            continue
        idx = int(meta["index"])
        score = float(row["Prediction"])
        out.loc[idx, "score"] = score
        out.loc[idx, "y_hat"] = 1 if score >= 0.5 else 0

    missing = int(out["score"].isna().sum())
    if missing:
        raise ValueError(f"{cfg.name}: missing predictions for {missing} staged samples.")
    if not args.keep_staging and staging_root.exists():
        shutil.rmtree(staging_root)
    return out


def _strip_module_prefix(state_dict: Dict[str, object]) -> Dict[str, object]:
    keys = list(state_dict.keys())
    if keys and all(k.startswith("module.") for k in keys):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _normalize_grad_uint8(grad_tensor) -> Image.Image:
    grad_np = grad_tensor.detach().cpu().permute(1, 2, 0).numpy()
    grad_np = grad_np - grad_np.min()
    max_val = float(grad_np.max())
    if max_val > 0:
        grad_np = grad_np / max_val
    grad_np = np.clip(grad_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(grad_np, mode="RGB")


def _load_lgrad_classifier_model(repo_root: Path, checkpoint_path: Path):
    import torch

    sys.path.insert(0, str(repo_root / "CNNDetection"))
    from networks.resnet import resnet50  # type: ignore

    model = resnet50(num_classes=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=False)
    return model


def _load_cnndetection_model(repo_root: Path, checkpoint_path: Path):
    import torch

    sys.path.insert(0, str(repo_root))
    from networks.resnet import resnet50  # type: ignore

    model = resnet50(num_classes=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=False)
    return model


def _load_lgrad_gradient_model(repo_root: Path, preprocessing_ckpt: Path):
    import torch

    sys.path.insert(0, str(repo_root / "img2gad_pytorch"))
    from models import build_model  # type: ignore

    model = build_model(
        gan_type="stylegan",
        module="discriminator",
        resolution=256,
        label_size=0,
        image_channels=3,
    )
    state = torch.load(preprocessing_ckpt, map_location="cpu")
    model.load_state_dict(state, strict=True)
    return model


def _load_f3net_model(repo_root: Path, checkpoint_path: Path):
    import torch

    sys.path.insert(0, str(repo_root))
    from PyDeepFakeDet.models.f3net import F3Net  # type: ignore

    model_cfg = {
        "IMG_WIDTH": 299,
        "IMG_HEIGHT": 299,
        "LFS_WINDOW_SIZE": 10,
        "LFS_M": 6,
        "XCEPTION_CFG": {"PRETRAINED": "", "ESCAPE": ""},
        "NUM_CLASSES": 2,
    }
    model = F3Net(model_cfg)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model_state", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[warn] F3Net missing keys: {len(missing)}")
    if unexpected:
        print(f"[warn] F3Net unexpected keys: {len(unexpected)}")
    return model


def _load_npr_model(repo_root: Path, checkpoint_path: Path):
    import torch

    sys.path.insert(0, str(repo_root))
    from networks.resnet import resnet50  # type: ignore

    model = resnet50(num_classes=1)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(ckpt, dict):
        state = ckpt.get("model", ckpt.get("state_dict", ckpt))
    else:
        state = ckpt
    state = _strip_module_prefix(state)
    model.load_state_dict(state, strict=True)
    return model


def run_f3net(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    repo_root = ensure_archive(PYDEEPFAKEDET_ARCHIVE_URL, args.external_root / "PyDeepFakeDet", "PyDeepFakeDet-main")
    pydeep_root = repo_root / "PyDeepFakeDet"
    if pydeep_root.exists():
        site.addsitedir(str(pydeep_root))
    site.addsitedir(str(repo_root))
    checkpoint = resolve_f3net_checkpoint(args)

    import torch
    from torchvision import transforms

    device = args.device
    model = _load_f3net_model(repo_root, checkpoint).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    paths = df["file_path"].astype(str).tolist()
    loader = build_loader(paths, transform, args)
    scores_by_path: Dict[str, float] = {}
    use_amp = amp_enabled(args)

    with torch.no_grad():
        for batch_paths, batch_tensors in tqdm(loader, desc="F3Net inference", leave=False):
            batch_tensors = batch_tensors.to(device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                outputs = model({"img": batch_tensors})
            logits = outputs["logits"]
            if logits.ndim != 2 or logits.shape[1] < 2:
                raise ValueError("Unexpected F3Net output format.")
            probs = torch.softmax(logits, dim=1)
            batch_scores = probs[:, 1].detach().cpu().numpy().astype(float).tolist()
            for path, score in zip(batch_paths, batch_scores):
                scores_by_path[str(path)] = float(score)

    out = df.copy()
    out["detector_name"] = "f3net"
    out["score"] = out["file_path"].astype(str).map(scores_by_path)
    out["y_hat"] = (out["score"] >= 0.5).astype(int)
    out["split"] = "test"
    return out


def run_npr_official(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    repo_root = ensure_archive(
        NPR_OFFICIAL_ARCHIVE_URL,
        args.external_root / "NPR-DeepfakeDetection",
        "NPR-DeepfakeDetection-main",
    )
    checkpoint = resolve_npr_checkpoint(args)

    import torch
    from torchvision import transforms

    device = args.device
    model = _load_npr_model(repo_root, checkpoint).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    paths = df["file_path"].astype(str).tolist()
    loader = build_loader(paths, transform, args, crop_even=True)
    scores_by_path: Dict[str, float] = {}
    use_amp = amp_enabled(args)

    with torch.no_grad():
        for batch_paths, batch_tensors in tqdm(loader, desc="NPR inference", leave=False):
            batch_tensors = batch_tensors.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                preds = model(batch_tensors).sigmoid().flatten()
            batch_scores = preds.detach().cpu().numpy().astype(float).tolist()
            for path, score in zip(batch_paths, batch_scores):
                scores_by_path[str(path)] = float(score)

    out = df.copy()
    out["detector_name"] = "npr"
    out["score"] = out["file_path"].astype(str).map(scores_by_path)
    out["y_hat"] = (out["score"] >= 0.5).astype(int)
    out["split"] = "test"
    return out


def run_lgrad_official(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    repo_root = ensure_archive(LGRAD_OFFICIAL_ARCHIVE_URL, args.external_root / "LGrad", "LGrad-master")
    checkpoint = resolve_lgrad_checkpoint(args)
    preprocessing_ckpt = resolve_lgrad_preprocessing_ckpt(args)

    import torch
    from torchvision import transforms

    device = args.device
    grad_model = _load_lgrad_gradient_model(repo_root, preprocessing_ckpt).to(device)
    cls_model = _load_lgrad_classifier_model(repo_root, checkpoint).to(device)
    grad_model.eval()
    cls_model.eval()

    grad_input_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    classifier_transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    paths = df["file_path"].astype(str).tolist()
    grad_loader = build_loader(paths, grad_input_transform, args)
    scores_by_path: Dict[str, float] = {}
    use_amp = amp_enabled(args)

    for batch_paths, grad_inputs in tqdm(grad_loader, desc="LGrad inference", leave=False):
        grad_inputs = grad_inputs.to(device=device, dtype=torch.float32)
        grad_inputs.requires_grad_(True)

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            pre = grad_model(grad_inputs)
        grad_model.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(
            pre.sum(),
            grad_inputs,
            create_graph=False,
            retain_graph=False,
            allow_unused=False,
        )[0]

        cls_inputs = []
        for sample_grad in grads:
            grad_img = _normalize_grad_uint8(sample_grad)
            cls_inputs.append(classifier_transform(grad_img))
        cls_batch = torch.stack(cls_inputs, dim=0).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                preds = cls_model(cls_batch).sigmoid().flatten()
        batch_scores = preds.detach().cpu().numpy().astype(float).tolist()
        for path, score in zip(batch_paths, batch_scores):
            scores_by_path[str(path)] = float(score)

    out = df.copy()
    out["detector_name"] = "lgrad"
    out["score"] = out["file_path"].astype(str).map(scores_by_path)
    out["y_hat"] = (out["score"] >= 0.5).astype(int)
    out["split"] = "test"
    return out


def run_cnndetection_official(df: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    repo_root = ensure_archive(
        CNNDETECTION_OFFICIAL_ARCHIVE_URL,
        args.external_root / "CNNDetection",
        "CNNDetection-master",
    )
    ensure_requirements(repo_root / "requirements.txt", args.external_root / ".installed" / "cnndetection.ok")
    checkpoint = resolve_cnndetection_checkpoint(args)

    import torch
    from torchvision import transforms

    device = args.device
    model = _load_cnndetection_model(repo_root, checkpoint).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    paths = df["file_path"].astype(str).tolist()
    loader = build_loader(paths, transform, args)
    scores_by_path: Dict[str, float] = {}
    use_amp = amp_enabled(args)

    with torch.no_grad():
        for batch_paths, batch_tensors in tqdm(loader, desc="CNNDetection inference", leave=False):
            batch_tensors = batch_tensors.to(device=device, dtype=torch.float32)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                preds = model(batch_tensors).sigmoid().flatten()
            batch_scores = preds.detach().cpu().numpy().astype(float).tolist()
            for path, score in zip(batch_paths, batch_scores):
                scores_by_path[str(path)] = float(score)

    out = df.copy()
    out["detector_name"] = "cnndetection"
    out["score"] = out["file_path"].astype(str).map(scores_by_path)
    out["y_hat"] = (out["score"] >= 0.5).astype(int)
    out["split"] = "test"
    return out


def main() -> None:
    args = parse_args()
    validate_runtime(args.device)
    ensure_dir(args.external_root)
    print(f"[info] running on device: {args.device}")
    df = load_input_dataframe(args)
    detector_names = parse_detector_names(args.detector)

    for detector_name in detector_names:
        run_args = argparse.Namespace(**vars(args))
        run_args.detector = detector_name
        cfg = DETECTOR_CONFIGS[detector_name]
        print(f"\n[info] running detector: {detector_name}")

        if cfg.backend == "sidbench":
            out = run_sidbench(df, run_args, cfg)
        elif cfg.backend == "cnndetection_official":
            out = run_cnndetection_official(df, run_args)
        elif cfg.backend == "lgrad_official":
            out = run_lgrad_official(df, run_args)
        elif cfg.backend == "npr_official":
            out = run_npr_official(df, run_args)
        elif cfg.backend == "pydeepfakedet_f3net":
            out = run_f3net(df, run_args)
        else:
            raise ValueError(f"Unsupported detector backend: {cfg.backend}")

        save_outputs(out, run_args)


if __name__ == "__main__":
    main()
