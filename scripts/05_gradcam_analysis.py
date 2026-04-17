"""
Stage 05: Model-based Grad-CAM attribution.

This script generates actual gradient-based activation maps from the pretrained
detectors used in Stage 03, instead of the previous residual-visualization
proxy.

Current thesis detector set:
- CNNDetection
- LGrad
- NPR

Legacy compatibility backends still available when explicitly requested via the
detector CSV:
- Gram via SIDBench-backed pretrained models
- F3Net via the PyDeepFakeDet pretrained F3Net checkpoint
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TARGET_LAYER_CANDIDATES = {
    "cnndetection": [
        "layer4.2.conv3",
        "layer4.1.conv3",
        "layer4.2.conv2",
        "model.layer4.2.conv3",
        "model.layer4.1.conv3",
        "model.layer4.2.conv2",
    ],
    "gram": [
        "model.layer4.2.conv3",
        "model.layer4.1.conv3",
        "model.layer4.2.conv2",
    ],
    "lgrad": [
        "layer4.2.conv3",
        "layer4.1.conv3",
        "layer4.2.conv2",
        "model.layer4.2.conv3",
        "model.layer4.1.conv3",
        "model.layer4.2.conv2",
    ],
    "npr": [
        "layer4.2.conv3",
        "layer4.1.conv3",
        "layer4.2.conv2",
        "model.layer4.2.conv3",
        "model.layer4.1.conv3",
        "model.layer4.2.conv2",
    ],
    "f3net": [
        "xception.conv4.conv1",
        "xception.conv3.conv1",
        "xception.conv2.conv1",
    ],
}

CURRENT_THESIS_DETECTORS = {"cnndetection", "lgrad", "npr"}
LEGACY_COMPATIBILITY_DETECTORS = {"gram", "f3net"}


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--detector-csv", type=Path, default=root / "results" / "detector_outputs" / "cnndetection_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=root / "results" / "attribution")
    parser.add_argument("--external-root", type=Path, default=root / ".external_models")
    parser.add_argument("--only-false-negative", action="store_true")
    parser.add_argument("--analyze-all", action="store_true",
                        help="Generate heatmaps for ALL samples, not just misclassified ones.")
    parser.add_argument("--max-per-group", type=int, default=10,
                        help="Max samples per group to generate heatmaps for (when --analyze-all).")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference device for Grad-CAM. Default is cuda.",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Enable AMP for compatible Grad-CAM forward paths on CUDA.",
    )
    return parser.parse_args()


def validate_runtime(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested for Grad-CAM, but no GPU is available. "
            "In Colab, switch Runtime -> Change runtime type -> GPU, "
            "or pass --device cpu if you intentionally want CPU mode."
        )


def load_stage03_module(project_root: Path):
    script_path = project_root / "scripts" / "03_run_detectors.py"
    spec = importlib.util.spec_from_file_location("stage03_detector_module", script_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Stage 03 module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module, device: str = "cuda", use_amp: bool = False) -> None:
        self.model = model
        self.target_layer = target_layer
        self.device = device
        self.use_amp = bool(use_amp and device == "cuda")
        self.activations = None
        self.gradients = None
        self._fwd_handle = target_layer.register_forward_hook(self._save_activation)
        self._bwd_handle = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output) -> None:
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output) -> None:
        self.gradients = grad_output[0].detach()

    def close(self) -> None:
        self._fwd_handle.remove()
        self._bwd_handle.remove()

    def generate(self, input_tensor: torch.Tensor, score_selector: Callable[[object], torch.Tensor]) -> Tuple[np.ndarray, float]:
        self.model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_amp):
            output = self.model(input_tensor)
        score = score_selector(output)
        if score.ndim > 0:
            score = score.reshape(-1)[0]
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0].detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)

        score_value = float(score.detach().cpu().item())
        return cam, score_value


def infer_detector_name(df: pd.DataFrame, detector_csv: Path) -> str:
    if "detector_name" in df.columns and len(df) > 0:
        name = str(df["detector_name"].iloc[0]).strip().lower()
        if name:
            return name
    stem = detector_csv.stem.lower()
    for name in ("cnndetection", "lgrad", "npr", "gram", "f3net"):
        if name in stem:
            return name
    raise ValueError(f"Could not infer detector name from {detector_csv}")


def select_target_df(df: pd.DataFrame, analyze_all: bool, only_false_negative: bool, max_per_group: int) -> pd.DataFrame:
    if analyze_all:
        target_df = df.copy()
        if max_per_group:
            parts = []
            for _, sub in target_df.groupby("group"):
                parts.append(sub.head(max_per_group))
            target_df = pd.concat(parts, ignore_index=True)
        return target_df

    mis = df[df["y_true"].astype(int) != df["y_hat"].astype(int)].copy()
    if only_false_negative:
        mis = mis[(mis["y_true"].astype(int) == 1) & (mis["y_hat"].astype(int) == 0)].copy()
    return mis.reset_index(drop=True)


def resolve_image_path(project_root: Path, path_value: str) -> Path:
    candidate = Path(str(path_value))
    if candidate.is_absolute():
        return candidate
    return (project_root / candidate).resolve()


def choose_target_layer(model: nn.Module, detector_name: str) -> nn.Module:
    named_modules = dict(model.named_modules())
    for layer_name in TARGET_LAYER_CANDIDATES.get(detector_name, []):
        layer = named_modules.get(layer_name)
        if isinstance(layer, nn.Conv2d):
            return layer
    conv_layers = [m for m in model.modules() if isinstance(m, nn.Conv2d)]
    if not conv_layers:
        raise ValueError("Could not find any Conv2d layer for Grad-CAM.")
    return conv_layers[-1]


def overlay_heatmap_on_image(image: Image.Image, heatmap: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    img_arr = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    color_map = cm.get_cmap("jet")
    heatmap_color = color_map(heatmap)[..., :3]
    overlay = 0.45 * img_arr + 0.55 * heatmap_color
    overlay = np.clip(overlay, 0.0, 1.0)
    return heatmap_color, overlay


def save_gradcam_triptych(
    image: Image.Image,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_path: Path,
    detector_name: str,
    score_value: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(image)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    axes[1].imshow(heatmap, cmap="jet")
    axes[1].set_title(f"{detector_name} Grad-CAM")
    axes[1].axis("off")

    axes[2].imshow(overlay)
    axes[2].set_title(f"Overlay (score={score_value:.4f})")
    axes[2].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_sidbench_components(stage03, detector_name: str, external_root: Path, device: str):
    cfg = stage03.DETECTOR_CONFIGS[detector_name]
    repo_root = stage03.ensure_archive(stage03.SIDBENCH_ARCHIVE_URL, external_root / "sidbench", "sidbench-main")
    stage03.ensure_requirements(repo_root / "requirements.txt", external_root / ".installed" / "sidbench.ok")
    weights_zip = stage03.download_gdrive(stage03.SIDBENCH_WEIGHTS_URL, external_root / "sidbench_weights.zip")
    weights_root = external_root / "sidbench_weights"
    if not (weights_root / ".ready").exists():
        stage03.ensure_dir(weights_root)
        import zipfile
        with zipfile.ZipFile(weights_zip, "r") as zf:
            zf.extractall(weights_root)
        (weights_root / ".ready").write_text("ok", encoding="utf-8")

    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from models.CNNDetect import CNNDetect  # type: ignore
    from models.GramNet import GramNet  # type: ignore
    from models.LGrad import LGrad  # type: ignore
    from preprocessing.lgrad.models import build_model  # type: ignore

    checkpoint = weights_root / cfg.sidbench_ckpt_relpath
    if detector_name == "cnndetection":
        model = CNNDetect()
        model.load_weights(str(checkpoint))
        load_size, crop_size = 256, 256
    elif detector_name == "gram":
        model = GramNet()
        model.load_weights(str(checkpoint))
        load_size, crop_size = 299, 299
    elif detector_name == "lgrad":
        gen_model = build_model(
            gan_type="stylegan",
            module="discriminator",
            resolution=256,
            label_size=0,
            image_channels=3,
        )
        gen_ckpt = weights_root / "weights" / "preprocessing" / "karras2019stylegan-bedrooms-256x256_discriminator.pth"
        gen_model.load_state_dict(torch.load(gen_ckpt, map_location="cpu"), strict=True)
        gen_model = gen_model.to(device).eval()
        model = LGrad()
        model.load_weights(str(checkpoint))
        load_size, crop_size = None, 256
    else:
        raise ValueError(f"Unsupported SIDBench detector for Grad-CAM: {detector_name}")

    model = model.to(device).eval()

    preprocess = build_sidbench_preprocess(detector_name, load_size, crop_size, gen_model if detector_name == "lgrad" else None)
    target_layer = choose_target_layer(model, detector_name)
    score_selector = sidbench_score_selector(detector_name)
    return model, preprocess, target_layer, score_selector, device


def build_official_preprocess(detector_name: str, lgrad_model: nn.Module | None = None):
    base_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    if detector_name != "lgrad":
        def preprocess(image: Image.Image, device: str) -> torch.Tensor:
            if detector_name == "npr":
                width, height = image.size
                if width % 2 == 1 or height % 2 == 1:
                    image = image.crop((0, 0, width - (width % 2), height - (height % 2)))
            return base_transform(image).unsqueeze(0).to(device)

        return preprocess

    if lgrad_model is None:
        raise ValueError("LGrad preprocessing requires the gradient model.")

    gen_transform = transforms.Compose(
        [
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    final_transform = transforms.Compose(
        [
            transforms.CenterCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    def preprocess(image: Image.Image, device: str) -> torch.Tensor:
        img_tensor = gen_transform(image).unsqueeze(0).to(device=device, dtype=torch.float32)
        img_tensor.requires_grad_(True)
        pre = lgrad_model(img_tensor)
        lgrad_model.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(pre.sum(), img_tensor, create_graph=False, retain_graph=False)[0]
        grad_pil = stage03_normalize_grad_uint8(grads[0])
        return final_transform(grad_pil).unsqueeze(0).to(device)

    return preprocess


def stage03_normalize_grad_uint8(grad_tensor) -> Image.Image:
    grad_np = grad_tensor.detach().cpu().permute(1, 2, 0).numpy()
    grad_np = grad_np - grad_np.min()
    max_val = float(grad_np.max())
    if max_val > 0:
        grad_np = grad_np / max_val
    grad_np = np.clip(grad_np * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(grad_np, mode="RGB")


def build_official_components(stage03, detector_name: str, external_root: Path, device: str):
    if detector_name == "cnndetection":
        repo_root = stage03.ensure_archive(
            stage03.CNNDETECTION_OFFICIAL_ARCHIVE_URL,
            external_root / "CNNDetection",
            "CNNDetection-master",
        )
        stage03.ensure_requirements(repo_root / "requirements.txt", external_root / ".installed" / "cnndetection.ok")
        checkpoint = stage03.resolve_cnndetection_checkpoint(
            argparse.Namespace(external_root=external_root)
        )
        model = stage03._load_cnndetection_model(repo_root, checkpoint).to(device).eval()
        preprocess = build_official_preprocess(detector_name)
    elif detector_name == "lgrad":
        repo_root = stage03.ensure_archive(
            stage03.LGRAD_OFFICIAL_ARCHIVE_URL,
            external_root / "LGrad",
            "LGrad-master",
        )
        checkpoint = stage03.resolve_lgrad_checkpoint(argparse.Namespace(external_root=external_root))
        preprocessing_ckpt = stage03.resolve_lgrad_preprocessing_ckpt(
            argparse.Namespace(external_root=external_root)
        )
        grad_model = stage03._load_lgrad_gradient_model(repo_root, preprocessing_ckpt).to(device).eval()
        model = stage03._load_lgrad_classifier_model(repo_root, checkpoint).to(device).eval()
        preprocess = build_official_preprocess(detector_name, lgrad_model=grad_model)
    elif detector_name == "npr":
        repo_root = stage03.ensure_archive(
            stage03.NPR_OFFICIAL_ARCHIVE_URL,
            external_root / "NPR-DeepfakeDetection",
            "NPR-DeepfakeDetection-main",
        )
        checkpoint = stage03.resolve_npr_checkpoint(argparse.Namespace(external_root=external_root))
        model = stage03._load_npr_model(repo_root, checkpoint).to(device).eval()
        preprocess = build_official_preprocess(detector_name)
    else:
        raise ValueError(f"Unsupported official detector for Grad-CAM: {detector_name}")

    target_layer = choose_target_layer(model, detector_name)

    def score_selector(output: object) -> torch.Tensor:
        logits = output["logits"] if isinstance(output, dict) and "logits" in output else output
        return logits.reshape(-1)[0]

    return model, preprocess, target_layer, score_selector, device


def build_sidbench_preprocess(detector_name: str, load_size: int | None, crop_size: int, lgrad_model: nn.Module | None):
    resize_ops: List[transforms.Compose] = []
    if load_size is not None:
        resize_ops.append(transforms.Resize((load_size, load_size)))
    resize_ops.append(transforms.CenterCrop(crop_size))

    if detector_name != "lgrad":
        base = transforms.Compose(
            resize_ops + [
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )

        def preprocess(image: Image.Image, device: str) -> torch.Tensor:
            return base(image).unsqueeze(0).to(device)

        return preprocess

    if lgrad_model is None:
        raise ValueError("LGrad preprocessing requires the generator/discriminator model.")

    gen_transform = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    final_transform = transforms.Compose(
        [
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )

    def normalize_np(img: np.ndarray) -> np.ndarray:
        img = img.copy()
        img -= img.min()
        maxv = img.max()
        if maxv != 0:
            img /= maxv
        return img * 255.0

    def preprocess(image: Image.Image, device: str) -> torch.Tensor:
        img_tensor = gen_transform(image).unsqueeze(0).to(device=device, dtype=torch.float32)
        img_tensor.requires_grad_(True)
        pre = lgrad_model(img_tensor)
        lgrad_model.zero_grad(set_to_none=True)
        grads = torch.autograd.grad(pre.sum(), img_tensor, create_graph=False, retain_graph=False)[0]
        grad_img = grads[0].permute(1, 2, 0).detach().cpu().numpy()
        grad_img = normalize_np(grad_img).astype(np.uint8)
        grad_pil = Image.fromarray(grad_img).convert("RGB")
        return final_transform(grad_pil).unsqueeze(0).to(device)

    return preprocess


def sidbench_score_selector(detector_name: str) -> Callable[[object], torch.Tensor]:
    def selector(output: object) -> torch.Tensor:
        if isinstance(output, dict) and "logits" in output:
            logits = output["logits"]
        else:
            logits = output
        logits = logits.reshape(-1)
        return logits[0]

    return selector


def build_f3net_components(stage03, external_root: Path, device: str):
    repo_root = stage03.ensure_archive(stage03.PYDEEPFAKEDET_ARCHIVE_URL, external_root / "PyDeepFakeDet", "PyDeepFakeDet-main")
    stage03.ensure_requirements(repo_root / "requirements.txt", external_root / ".installed" / "pydeepfakedet.ok")
    checkpoint = stage03.download_gdrive(stage03.F3NET_RAW_CKPT_URL, external_root / "weights" / "f3net_raw.pth")

    model = stage03._load_f3net_model(repo_root, checkpoint).to(device).eval()
    target_layer = choose_target_layer(model, "f3net")

    preprocess = transforms.Compose(
        [
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def preprocess_fn(image: Image.Image, device_name: str) -> torch.Tensor:
        return preprocess(image).unsqueeze(0).to(device_name)

    def score_selector(output: object) -> torch.Tensor:
        if not isinstance(output, dict) or "logits" not in output:
            raise ValueError("Unexpected F3Net output format for Grad-CAM.")
        logits = output["logits"]
        if logits.ndim != 2 or logits.shape[1] < 2:
            raise ValueError("Unexpected F3Net logits shape.")
        return logits[:, 1]

    return model, preprocess_fn, target_layer, score_selector, device


def load_detector_components(detector_name: str, project_root: Path, external_root: Path, device: str):
    stage03 = load_stage03_module(project_root)
    if detector_name in {"cnndetection", "lgrad", "npr"}:
        return build_official_components(stage03, detector_name, external_root, device)
    if detector_name in {"gram"}:
        return build_sidbench_components(stage03, detector_name, external_root, device)
    if detector_name == "f3net":
        return build_f3net_components(stage03, external_root, device)
    raise ValueError(f"Unsupported detector for Grad-CAM: {detector_name}")


def main() -> None:
    args = parse_args()
    validate_runtime(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    heatmap_dir = args.output_dir / "heatmaps"
    heatmap_dir.mkdir(parents=True, exist_ok=True)
    print(f"[info] running on device: {args.device}")

    df = pd.read_csv(args.detector_csv)
    detector_name = infer_detector_name(df, args.detector_csv)
    if detector_name in LEGACY_COMPATIBILITY_DETECTORS:
        print(
            f"[warn] detector={detector_name} is kept only as a legacy compatibility backend "
            "and is not part of the current thesis mainline."
        )
    target_df = select_target_df(df, args.analyze_all, args.only_false_negative, args.max_per_group)
    target_df.to_csv(args.output_dir / "analyzed_samples.csv", index=False, encoding="utf-8")

    print(f"Generating Grad-CAM for detector={detector_name}, samples={len(target_df)}, amp={args.amp}")
    if len(target_df) == 0:
        note = args.output_dir / "README_GradCAM.txt"
        note.write_text("No samples selected for Grad-CAM analysis.", encoding="utf-8")
        return

    model, preprocess_fn, target_layer, score_selector, device = load_detector_components(
        detector_name, args.project_root, args.external_root, args.device
    )
    gradcam = GradCAM(model, target_layer, device=device, use_amp=args.amp and detector_name in {"cnndetection", "npr", "f3net"})

    try:
        for _, row in target_df.iterrows():
            image_path = resolve_image_path(args.project_root, str(row["file_path"]))
            if not image_path.exists():
                print(f"[warn] Missing file: {image_path}")
                continue

            image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess_fn(image, device)
            cam, score_value = gradcam.generate(input_tensor, score_selector)

            cropped_image = transforms.CenterCrop(cam.shape[0])(image)
            heatmap_color, overlay = overlay_heatmap_on_image(cropped_image, cam)
            group = row.get("group", "unknown")
            actual = "Fake" if int(row["y_true"]) == 1 else "Real"
            sid = image_path.stem
            out_name = f"{group}_Actual{actual}_{sid}_gradcam.png"
            save_gradcam_triptych(
                cropped_image,
                cam,
                overlay,
                heatmap_dir / out_name,
                detector_name,
                score_value,
            )
    finally:
        gradcam.close()

    note = args.output_dir / "README_GradCAM.txt"
    note.write_text(
        f"Grad-CAM attribution completed with detector={detector_name}. "
        f"Check the 'heatmaps' folder for model-based visualizations.",
        encoding="utf-8",
    )
    print(f"[saved] heatmaps: {heatmap_dir}")


if __name__ == "__main__":
    main()
