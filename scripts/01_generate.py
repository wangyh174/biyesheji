"""
Step 1: Generate candidate synthetic samples for fairness evaluation.

Official thesis mode:
1) Fair-Diffusion for fake-image generation
2) Local real-image registration for control samples
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=root)
    parser.add_argument("--metadata-out", type=Path, default=root / "data" / "metadata_raw.csv")
    parser.add_argument("--samples-per-group", type=int, default=50)
    parser.add_argument(
        "--real-per-group",
        type=int,
        default=None,
        help="How many real samples to register per group. For --real-source local, uses all available files when omitted.",
    )
    parser.add_argument(
        "--mock-real-per-group",
        type=int,
        default=None,
        help="Deprecated alias of --real-per-group. Kept only for backward compatibility.",
    )
    parser.add_argument("--genders", type=str, default="male,female")
    parser.add_argument("--professions", type=str, default="doctor,nurse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--generator",
        type=str,
        choices=["fairdiffusion"],
        default="fairdiffusion",
        help="Official thesis pipeline only supports Fair-Diffusion generation.",
    )
    parser.add_argument(
        "--real-source",
        type=str,
        choices=["local"],
        default="local",
        help="Official thesis pipeline only supports local real-image registration.",
    )
    parser.add_argument("--model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--real-model-id", type=str, default="runwayml/stable-diffusion-v1-5")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional local path for fake-image generation model (overrides --model-id if provided).",
    )
    parser.add_argument(
        "--real-model-path",
        type=Path,
        default=None,
        help="Optional local path for real-source diffusers model (overrides --real-model-id if provided).",
    )
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for fake image generation. Increase on high-VRAM GPUs.",
    )
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument(
        "--torch-dtype",
        type=str,
        choices=["auto", "float16", "bfloat16", "float32"],
        default="float16",
        help="Torch dtype for diffusion pipeline weights. Default is float16 for faster CUDA generation.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cpu", "cuda"],
        help="Inference device. Default is cuda; the script will fail fast if CUDA is unavailable.",
    )
    parser.add_argument(
        "--enable-xformers",
        action="store_true",
        help="Enable xFormers memory-efficient attention when available.",
    )
    parser.add_argument(
        "--enable-vae-slicing",
        action="store_true",
        default=False,
        help="Enable VAE slicing to reduce memory pressure.",
    )
    parser.add_argument(
        "--disable-vae-slicing",
        dest="enable_vae_slicing",
        action="store_false",
        help="Disable VAE slicing.",
    )
    parser.add_argument(
        "--enable-vae-tiling",
        action="store_true",
        help="Enable VAE tiling for larger images or tighter VRAM budgets.",
    )
    parser.add_argument("--negative-prompt", type=str, default="blurry, low quality, distorted, bad anatomy, deformed eyes, crossed eyes, disfigured, poorly drawn face, ugly, cartoon, plastic, artificial, weird proportions, fake")
    # Fair-Diffusion style controls (from official README usage).
    parser.add_argument("--fd-editing-prompts", type=str, default="male person,female person")
    parser.add_argument("--fd-reverse-directions", type=str, default="true,false")
    parser.add_argument("--fd-warmup-steps", type=str, default="10,10")
    parser.add_argument("--fd-guidance-scales", type=str, default="4,4")
    parser.add_argument("--fd-thresholds", type=str, default="0.95,0.95")
    parser.add_argument("--fd-weights", type=str, default="1,1")
    parser.add_argument("--fd-momentum-scale", type=float, default=0.3)
    parser.add_argument("--fd-momentum-beta", type=float, default=0.6)
    parser.add_argument("--fd-female-prob", type=float, default=0.5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_runtime(device: str) -> None:
    if device == "cuda":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA device was requested, but no GPU is available. "
                "In Colab, switch Runtime -> Change runtime type -> GPU, "
                "or pass --device cpu if you intentionally want CPU mode."
            )


def resolve_torch_dtype(device: str, torch_dtype: str):
    import torch

    if torch_dtype == "float16":
        return torch.float16
    if torch_dtype == "bfloat16":
        return torch.bfloat16
    if torch_dtype == "float32":
        return torch.float32

    if device == "cuda":
        return torch.float16
    return torch.float32


def optimize_pipeline(pipe, device: str, enable_xformers: bool, enable_vae_slicing: bool, enable_vae_tiling: bool):
    if device == "cpu":
        pipe.enable_attention_slicing()
        return pipe

    # channels_last is a safe low-effort optimization for convolution-heavy pipelines.
    try:
        import torch

        if hasattr(pipe, "unet") and pipe.unet is not None:
            pipe.unet.to(memory_format=torch.channels_last)
        if hasattr(pipe, "vae") and pipe.vae is not None:
            pipe.vae.to(memory_format=torch.channels_last)
    except Exception:
        pass

    if enable_xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
            print("[opt] enabled xFormers memory efficient attention")
        except Exception as e:
            print(f"[warn] could not enable xFormers: {e}")

    if enable_vae_slicing:
        try:
            pipe.enable_vae_slicing()
            print("[opt] enabled VAE slicing")
        except Exception as e:
            print(f"[warn] could not enable VAE slicing: {e}")

    if enable_vae_tiling:
        try:
            pipe.enable_vae_tiling()
            print("[opt] enabled VAE tiling")
        except Exception as e:
            print(f"[warn] could not enable VAE tiling: {e}")

    return pipe


def resolve_real_per_group(args: argparse.Namespace) -> int | None:
    if args.real_per_group is not None and args.mock_real_per_group is not None:
        raise ValueError("Use only one of --real-per-group or --mock-real-per-group, not both.")
    if args.real_per_group is not None:
        return args.real_per_group
    if args.mock_real_per_group is not None:
        return args.mock_real_per_group
    return None


def list_local_real_files(group_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp"):
        files.extend(sorted(group_dir.glob(ext)))
    return files


def build_group_prompt(template: str, gender: str, profession: str) -> str:
    gender_desc = {
        "male": "adult male, man, masculine facial features",
        "female": "adult female, woman, feminine facial features",
    }.get(gender, gender)
    profession_desc = {
        "doctor": "doctor, physician, white coat, clinical badge, stethoscope",
        "nurse": "nurse, hospital nurse, medical scrubs, clinical ward",
    }.get(profession, profession)
    base = template.format(gender=gender, profession=profession)
    constraints = (
        "single person, realistic human face, natural skin texture, symmetrical eyes, "
        "intact facial features, professional medical environment, photojournalistic, high detail"
    )
    return f"{base} {gender_desc}, {profession_desc}, {constraints}."


def build_fairdiffusion_base_prompt(profession: str) -> str:
    profession_desc = {
        "doctor": "doctor, physician, white coat, stethoscope",
        "nurse": "nurse, hospital nurse, medical scrubs",
    }.get(profession, profession)
    return (
        f"A portrait photo of the face of a {profession}. "
        f"Hospital setting, {profession_desc}, realistic human, clinical identity visible."
    )


def build_group_negative_prompt(base_negative: str, gender: str, profession: str) -> str:
    parts = [base_negative]
    if gender == "male":
        parts.append("woman, female, feminine face, lipstick, heavy makeup, long eyelashes")
    elif gender == "female":
        parts.append("man, male, beard, mustache, masculine face")
    if profession == "doctor":
        parts.append("nurse cap, nurse-only uniform, patient gown")
    elif profession == "nurse":
        parts.append("doctor white coat only, physician portrait")
    parts.append(
        "two people, duplicate face, extra eyes, extra fingers, warped face, broken face, melted face, "
        "mask-like skin, doll, toy, figurine, cartoon, illustration, 3d render"
    )
    return ", ".join(x for x in parts if x)


def build_fairdiffusion_edit_config(gender: str) -> Tuple[List[str], List[bool]]:
    # Fair-Diffusion paper steers between "female person" and "male person"
    # while keeping the occupation prompt itself simple.
    editing_prompts = ["female person", "male person"]
    if gender == "female":
        # Toward female, away from male.
        reverse_dirs = [False, True]
    else:
        # Toward male, away from female.
        reverse_dirs = [True, False]
    return editing_prompts, reverse_dirs


def ensure_csv(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "id",
                "modality",
                "group",
                "gender",
                "profession",
                "prompt",
                "seed",
                "source_model",
                "source_domain",
                "y_true",
                "clip_score",
                "quality_score",
                "template_id",
                "file_path",
            ]
        )


def _rng_from_text(seed: int, text: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{seed}|{text}".encode("utf-8")).hexdigest()
    mixed = int(digest[:16], 16) % (2**32 - 1)
    return np.random.default_rng(mixed)


def make_mock_image(prompt: str, seed: int, width: int, height: int, kind: str) -> Image.Image:
    rng = _rng_from_text(seed, prompt + "|" + kind)
    if kind == "fake":
        arr = rng.integers(0, 255, size=(height, width, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB").filter(ImageFilter.EDGE_ENHANCE_MORE)
    else:
        # Simulate smoother natural texture for "real" references.
        arr = rng.normal(loc=140, scale=25, size=(height, width, 3)).clip(0, 255).astype(np.uint8)
        img = Image.fromarray(arr, mode="RGB").filter(ImageFilter.GaussianBlur(radius=1.2))

    draw = ImageDraw.Draw(img)
    draw.text((10, 10), prompt[:80], fill=(255, 255, 255))
    return img


def init_diffusers(
    model_id: str,
    device: str,
    torch_dtype: str,
    enable_xformers: bool,
    enable_vae_slicing: bool,
    enable_vae_tiling: bool,
):
    from diffusers import StableDiffusionPipeline

    dtype = resolve_torch_dtype(device, torch_dtype)
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe = optimize_pipeline(
        pipe,
        device=device,
        enable_xformers=enable_xformers,
        enable_vae_slicing=enable_vae_slicing,
        enable_vae_tiling=enable_vae_tiling,
    )
    return pipe, device


def _parse_csv_str(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_csv_float(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def _parse_csv_bool(s: str) -> List[bool]:
    out = []
    for x in s.split(","):
        t = x.strip().lower()
        if t in ("true", "1", "yes"):
            out.append(True)
        elif t in ("false", "0", "no"):
            out.append(False)
        elif t:
            raise ValueError(f"Invalid boolean value: {x}")
    return out


def init_fairdiffusion(
    model_id: str,
    device: str,
    torch_dtype: str,
    enable_xformers: bool,
    enable_vae_slicing: bool,
    enable_vae_tiling: bool,
):
    backend = None
    pipeline_cls = None

    try:
        from diffusers import SemanticStableDiffusionPipeline

        pipeline_cls = SemanticStableDiffusionPipeline
        backend = "diffusers.SemanticStableDiffusionPipeline"
    except Exception:
        try:
            from semdiffusers import SemanticEditPipeline

            pipeline_cls = SemanticEditPipeline
            backend = "semdiffusers.SemanticEditPipeline"
        except ImportError as e:
            raise RuntimeError(
                "Fair-Diffusion mode requires either "
                "`diffusers.SemanticStableDiffusionPipeline` or local `semdiffusers`. "
                "Recommended fix in Colab: use the official diffusers pipeline."
            ) from e

    dtype = resolve_torch_dtype(device, torch_dtype)

    pipe = pipeline_cls.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    pipe = optimize_pipeline(
        pipe,
        device=device,
        enable_xformers=enable_xformers,
        enable_vae_slicing=enable_vae_slicing,
        enable_vae_tiling=enable_vae_tiling,
    )
    return pipe, device, backend


def resolve_model_source(model_id: str, model_path: Path | None) -> str:
    """
    Resolve pretrained source for diffusers from local path or model id.
    Priority:
    1) explicit --model-path / --real-model-path
    2) when --model-id itself is a valid local directory, use it as local path
    3) fallback to remote model id
    """
    if model_path is not None:
        p = Path(model_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"Local model path does not exist: {p}")
        return str(p)

    id_as_path = Path(model_id).expanduser()
    if id_as_path.exists():
        return str(id_as_path.resolve())
    return model_id


def make_diffusers_image(
    pipe,
    device: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
) -> Image.Image:
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        generator=generator,
    )
    return out.images[0]


def make_fairdiffusion_images(
    pipe,
    device: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seeds: List[int],
    editing_prompts: List[str],
    reverse_dirs: List[bool],
    warmup_steps: List[int],
    edit_guidance_scales: List[float],
    thresholds: List[float],
    weights: List[float],
    momentum_scale: float,
    momentum_beta: float,
) -> List[Image.Image]:
    import torch

    prompts = [prompt] * len(seeds)
    negative_prompts = [negative_prompt] * len(seeds)
    generators = [torch.Generator(device=device).manual_seed(seed) for seed in seeds]
    out = pipe(
        prompt=prompts,
        negative_prompt=negative_prompts,
        generator=generators,
        num_inference_steps=steps,
        guidance_scale=guidance,
        width=width,
        height=height,
        editing_prompt=editing_prompts,
        reverse_editing_direction=reverse_dirs,
        edit_warmup_steps=warmup_steps,
        edit_guidance_scale=edit_guidance_scales,
        edit_threshold=thresholds,
        edit_momentum_scale=momentum_scale,
        edit_mom_beta=momentum_beta,
        edit_weights=weights,
    )
    return out.images


def append_rows(path: Path, rows: List[Dict[str, object]]) -> None:
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "id",
                "modality",
                "group",
                "gender",
                "profession",
                "prompt",
                "seed",
                "source_model",
                "source_domain",
                "y_true",
                "clip_score",
                "quality_score",
                "template_id",
                "file_path",
            ],
        )
        writer.writerows(rows)


def build_groups(genders_csv: str, professions_csv: str) -> List[Tuple[str, str, str]]:
    genders = [x.strip() for x in genders_csv.split(",") if x.strip()]
    professions = [x.strip() for x in professions_csv.split(",") if x.strip()]
    out = []
    for g in genders:
        for p in professions:
            out.append((f"{g}-{p}", g, p))
    return out


def sample_gender_for_profession(
    rng: random.Random,
    female_prob: float,
    remaining: Dict[str, int],
) -> str:
    if remaining["female"] <= 0:
        return "male"
    if remaining["male"] <= 0:
        return "female"
    return "female" if rng.random() < female_prob else "male"


def main() -> None:
    args = parse_args()
    if args.generator != "fairdiffusion":
        raise ValueError("This project is configured to run only with --generator fairdiffusion.")
    if args.real_source != "local":
        raise ValueError("This project is configured to run only with --real-source local.")
    validate_runtime(args.device)
    real_per_group = resolve_real_per_group(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    print(f"[info] running on device: {args.device}")

    root = args.project_root
    raw_dir = root / "data" / "generated_raw"
    real_dir = root / "data" / "real_samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    groups = build_groups(args.genders, args.professions)
    ensure_csv(args.metadata_out, overwrite=args.overwrite)

    pipe, device = (None, None)
    fake_source = resolve_model_source(args.model_id, args.model_path)
    pipe, device, fair_backend = init_fairdiffusion(
        fake_source,
        args.device,
        args.torch_dtype,
        args.enable_xformers,
        args.enable_vae_slicing,
        args.enable_vae_tiling,
    )
    print(
        f"[generator] fairdiffusion model={fake_source} device={device} "
        f"backend={fair_backend} dtype={args.torch_dtype}"
    )

    local_real_dir = Path(args.project_root) / "data" / "real_samples"
    print(f"[real-source] local directory: {local_real_dir}")

    rows: List[Dict[str, object]] = []
    fd_editing_prompts = _parse_csv_str(args.fd_editing_prompts)
    fd_reverse_dirs = _parse_csv_bool(args.fd_reverse_directions)
    fd_warmup_steps = _parse_csv_int(args.fd_warmup_steps)
    fd_guidance_scales = _parse_csv_float(args.fd_guidance_scales)
    fd_thresholds = _parse_csv_float(args.fd_thresholds)
    fd_weights = _parse_csv_float(args.fd_weights)

    print(
        f"[generate] width={args.width} height={args.height} steps={args.steps} "
        f"batch_size={args.batch_size}"
    )

    for group, gender, profession in groups:
        group_raw = raw_dir / group
        group_real = real_dir / group
        group_raw.mkdir(parents=True, exist_ok=True)
        group_real.mkdir(parents=True, exist_ok=True)

        prompt = build_fairdiffusion_base_prompt(profession)
        negative_prompt = build_group_negative_prompt(args.negative_prompt, gender=gender, profession=profession)
        image_editing_prompts, image_reverse_dirs = build_fairdiffusion_edit_config(gender)

        generated = 0
        while generated < args.samples_per_group:
            batch_n = min(args.batch_size, args.samples_per_group - generated)
            seeds = [args.seed + generated + j for j in range(batch_n)]
            start_idx = generated

            print(
                f"[stage] generating group={group} profession={profession} gender={gender} "
                f"start={start_idx} batch={batch_n}"
            )

            images = make_fairdiffusion_images(
                pipe=pipe,
                device=device,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=args.width,
                height=args.height,
                steps=args.steps,
                guidance=args.guidance,
                seeds=seeds,
                editing_prompts=image_editing_prompts,
                reverse_dirs=image_reverse_dirs,
                warmup_steps=fd_warmup_steps,
                edit_guidance_scales=fd_guidance_scales,
                thresholds=fd_thresholds,
                weights=fd_weights,
                momentum_scale=args.fd_momentum_scale,
                momentum_beta=args.fd_momentum_beta,
            )

            for j, img in enumerate(images):
                i = start_idx + j
                sid = f"fake_{group}_{i:04d}"
                seed_i = seeds[j]
                image_path = group_raw / f"{sid}.png"
                img.save(image_path)

                rows.append(
                    {
                        "id": sid,
                        "modality": "image",
                        "group": group,
                        "gender": gender,
                        "profession": profession,
                        "prompt": prompt,
                        "seed": seed_i,
                        "source_model": fake_source,
                        "source_domain": "generated",
                        "y_true": 1,
                        "clip_score": "",
                        "quality_score": "",
                        "template_id": -1,
                        "file_path": str(image_path),
                    }
                )
            generated += batch_n
        print(f"[done] group={group} fake={generated}")

    # Create/register "real" control samples (same semantic groups) for fairness evaluation.
    for group, gender, profession in groups:
        registered_real = 0
        if args.real_source == "local":
            local_group_dir = Path(args.project_root) / "data" / "real_samples" / group
            local_files = list_local_real_files(local_group_dir)
            if not local_files:
                print(f"  [warn] No local real images found for {group}: {local_group_dir}")
                continue

            use_files = local_files if real_per_group is None else local_files[:real_per_group]
            if real_per_group is not None and len(local_files) < real_per_group:
                print(f"  [warn] Local real images for {group}: have {len(local_files)}, requested {real_per_group}")

            for i, image_path in enumerate(use_files):
                prompt = f"A real photo of a {gender} {profession} in a hospital, realistic and natural."
                sid = f"real_{group}_{i:04d}"
                seed_i = args.seed + 10_000 + i
                rows.append(
                    {
                        "id": sid,
                        "modality": "image",
                        "group": group,
                        "gender": gender,
                        "profession": profession,
                        "prompt": prompt,
                        "seed": seed_i,
                        "source_model": "real_photograph",
                        "source_domain": "real_reference",
                        "y_true": 0,
                        "clip_score": "",
                        "quality_score": "",
                        "template_id": -1,
                        "file_path": str(image_path),
                    }
                )
                registered_real += 1
        print(f"[done] group={group} real={registered_real}")

    append_rows(args.metadata_out, rows)
    print(f"[saved] metadata: {args.metadata_out} rows={len(rows)}")


if __name__ == "__main__":
    main()
