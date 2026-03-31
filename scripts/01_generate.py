"""
Step 1: Generate candidate synthetic samples for fairness evaluation.

Supports:
1) mock generator (fast, for pipeline debugging)
2) diffusers generator (real text-to-image generation)
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
    parser.add_argument("--prompt-file", type=Path, default=root / "data" / "prompts" / "prompt_templates.txt")
    parser.add_argument("--metadata-out", type=Path, default=root / "data" / "metadata_raw.csv")
    parser.add_argument("--samples-per-group", type=int, default=50)
    parser.add_argument("--mock-real-per-group", type=int, default=50)
    parser.add_argument("--genders", type=str, default="male,female")
    parser.add_argument("--professions", type=str, default="doctor,nurse")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generator", type=str, choices=["mock", "diffusers", "fairdiffusion"], default="mock")
    parser.add_argument("--real-source", type=str, choices=["mock", "diffusers", "local"], default="mock")
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
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
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


def read_templates(path: Path) -> List[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line]


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


def init_diffusers(model_id: str):
    import torch
    from diffusers import StableDiffusionPipeline

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id, 
        torch_dtype=dtype,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if device == "cpu":
        pipe.enable_attention_slicing()
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


def init_fairdiffusion(model_id: str):
    import torch

    try:
        from semdiffusers import SemanticEditPipeline
    except ImportError as e:
        raise RuntimeError(
            "Fair-Diffusion mode requires semdiffusers. "
            "Install: pip install -e ./semantic-image-editing-main/semantic-image-editing-main"
        ) from e

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipe = SemanticEditPipeline.from_pretrained(
        model_id,
        safety_checker=None,
        feature_extractor=None,
        requires_safety_checker=False,
    ).to(device)
    if device == "cpu":
        pipe.enable_attention_slicing()
    return pipe, device


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


def make_fairdiffusion_image(
    pipe,
    device: str,
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance: float,
    seed: int,
    editing_prompts: List[str],
    reverse_dirs: List[bool],
    warmup_steps: List[int],
    edit_guidance_scales: List[float],
    thresholds: List[float],
    weights: List[float],
    momentum_scale: float,
    momentum_beta: float,
) -> Image.Image:
    import torch

    generator = torch.Generator(device=device).manual_seed(seed)
    out = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
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
    return out.images[0]


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
    random.seed(args.seed)
    np.random.seed(args.seed)

    root = args.project_root
    raw_dir = root / "data" / "generated_raw"
    real_dir = root / "data" / "real_samples"
    raw_dir.mkdir(parents=True, exist_ok=True)
    real_dir.mkdir(parents=True, exist_ok=True)

    templates = read_templates(args.prompt_file)
    groups = build_groups(args.genders, args.professions)
    ensure_csv(args.metadata_out, overwrite=args.overwrite)

    pipe, device = (None, None)
    real_pipe, real_device = (None, None)
    fake_source = resolve_model_source(args.model_id, args.model_path)
    real_source = resolve_model_source(args.real_model_id, args.real_model_path)

    if args.generator == "diffusers":
        pipe, device = init_diffusers(fake_source)
        print(f"[generator] diffusers model={fake_source} device={device}")
    elif args.generator == "fairdiffusion":
        pipe, device = init_fairdiffusion(fake_source)
        print(f"[generator] fairdiffusion model={fake_source} device={device}")
    else:
        print("[generator] mock (fast debug mode)")

    if args.real_source == "diffusers":
        if args.generator == "diffusers" and real_source == fake_source:
            real_pipe, real_device = pipe, device
        else:
            real_pipe, real_device = init_diffusers(real_source)
        print(f"[real-source] diffusers model={real_source} device={real_device}")
    elif args.real_source == "local":
        local_real_dir = Path(args.project_root) / "data" / "real_samples"
        print(f"[real-source] local directory: {local_real_dir}")
    else:
        print("[real-source] mock")

    rows: List[Dict[str, object]] = []
    fd_editing_prompts = _parse_csv_str(args.fd_editing_prompts)
    fd_reverse_dirs = _parse_csv_bool(args.fd_reverse_directions)
    fd_warmup_steps = _parse_csv_int(args.fd_warmup_steps)
    fd_guidance_scales = _parse_csv_float(args.fd_guidance_scales)
    fd_thresholds = _parse_csv_float(args.fd_thresholds)
    fd_weights = _parse_csv_float(args.fd_weights)

    if args.generator == "fairdiffusion":
        for profession in sorted({profession for _, _, profession in groups}):
            remaining = {gender: args.samples_per_group for gender in ["male", "female"]}
            counters = {gender: 0 for gender in ["male", "female"]}
            while remaining["male"] > 0 or remaining["female"] > 0:
                gender = sample_gender_for_profession(random, args.fd_female_prob, remaining)
                group = f"{gender}-{profession}"
                group_raw = raw_dir / group
                group_real = real_dir / group
                group_raw.mkdir(parents=True, exist_ok=True)
                group_real.mkdir(parents=True, exist_ok=True)

                i = counters[gender]
                t_id = i % len(templates)
                prompt = build_fairdiffusion_base_prompt(profession)
                negative_prompt = build_group_negative_prompt(args.negative_prompt, gender=gender, profession=profession)
                image_editing_prompts, image_reverse_dirs = build_fairdiffusion_edit_config(gender)
                sid = f"fake_{group}_{i:04d}"
                seed_i = args.seed + sum(counters.values())
                image_path = group_raw / f"{sid}.png"
                img = make_fairdiffusion_image(
                    pipe=pipe,
                    device=device,
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    guidance=args.guidance,
                    seed=seed_i,
                    editing_prompts=image_editing_prompts,
                    reverse_dirs=image_reverse_dirs,
                    warmup_steps=fd_warmup_steps,
                    edit_guidance_scales=fd_guidance_scales,
                    thresholds=fd_thresholds,
                    weights=fd_weights,
                    momentum_scale=args.fd_momentum_scale,
                    momentum_beta=args.fd_momentum_beta,
                )
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
                        "template_id": t_id,
                        "file_path": str(image_path),
                    }
                )
                counters[gender] += 1
                remaining[gender] -= 1
            print(
                f"[done] profession={profession} fake_male={counters['male']} "
                f"fake_female={counters['female']}"
            )
    else:
        for group, gender, profession in groups:
            group_raw = raw_dir / group
            group_real = real_dir / group
            group_raw.mkdir(parents=True, exist_ok=True)
            group_real.mkdir(parents=True, exist_ok=True)

            for i in range(args.samples_per_group):
                t_id = i % len(templates)
                prompt = build_group_prompt(templates[t_id], gender=gender, profession=profession)
                negative_prompt = build_group_negative_prompt(args.negative_prompt, gender=gender, profession=profession)
                sid = f"fake_{group}_{i:04d}"
                seed_i = args.seed + i
                image_path = group_raw / f"{sid}.png"
                if args.generator == "mock":
                    img = make_mock_image(prompt, seed_i, args.width, args.height, kind="fake")
                else:
                    img = make_diffusers_image(
                        pipe=pipe,
                        device=device,
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=args.width,
                        height=args.height,
                        steps=args.steps,
                        guidance=args.guidance,
                        seed=seed_i,
                    )
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
                        "source_model": fake_source if args.generator == "diffusers" else "mock_generator",
                        "source_domain": "generated",
                        "y_true": 1,
                        "clip_score": "",
                        "quality_score": "",
                        "template_id": t_id,
                        "file_path": str(image_path),
                    }
                )

    # Create "real" control samples (same semantic groups) for fairness evaluation.
    for group, gender, profession in groups:
        group_real = real_dir / group
        group_real.mkdir(parents=True, exist_ok=True)
        for i in range(args.mock_real_per_group):
            prompt = f"A real photo of a {gender} {profession} in a hospital, realistic and natural."
            sid = f"real_{group}_{i:04d}"
            seed_i = args.seed + 10_000 + i
            image_path = group_real / f"{sid}.png"
            if args.real_source == "local":
                # Use pre-downloaded real photos from data/real_samples/
                local_group_dir = Path(args.project_root) / "data" / "real_samples" / group
                local_files = sorted(local_group_dir.glob("*.png")) + sorted(local_group_dir.glob("*.jpg"))
                if i < len(local_files):
                    img = Image.open(local_files[i]).convert("RGB")
                    img = img.resize((args.width, args.height), Image.LANCZOS)
                else:
                    print(f"  [warn] Not enough local real images for {group}, have {len(local_files)}, need index {i}")
                    continue
            elif args.real_source == "diffusers":
                img = make_diffusers_image(
                    pipe=real_pipe,
                    device=real_device,
                    prompt=prompt,
                    negative_prompt=args.negative_prompt,
                    width=args.width,
                    height=args.height,
                    steps=args.steps,
                    guidance=args.guidance,
                    seed=seed_i,
                )
            else:
                img = make_mock_image(prompt, seed_i, args.width, args.height, kind="real")
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
                    "source_model": "real_photograph" if args.real_source == "local" else (real_source if args.real_source == "diffusers" else "real_mock"),
                    "source_domain": "real_reference",
                    "y_true": 0,
                    "clip_score": "",
                    "quality_score": "",
                    "template_id": -1,
                        "file_path": str(image_path),
                    }
                )
        print(f"[done] group={group} real={args.mock_real_per_group}")

    append_rows(args.metadata_out, rows)
    print(f"[saved] metadata: {args.metadata_out} rows={len(rows)}")


if __name__ == "__main__":
    main()
