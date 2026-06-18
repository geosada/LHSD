#!/usr/bin/env python

import argparse
from pathlib import Path

import torch
import torch.multiprocessing as mp
from diffusers import DDPMPipeline
from tqdm import tqdm
from PIL import Image
import numpy as np


# ---------- Optional: reuse your cache loader ----------
def load_ddpm_cache(out_dir, num_samples, start_index=0):
    """
    Load [num_samples] images starting from index [start_index].
    Expects files named ddpm_cifar10_{idx:05d}.png.
    Returns a tensor [num_samples, 3, 32, 32] in float32 [0,1].
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return None

    imgs = []
    for idx in range(start_index, start_index + num_samples):
        f = out_dir / f"ddpm_cifar10_{idx:05d}.png"
        if not f.exists():
            return None
        img = Image.open(f).convert("RGB")
        arr = np.array(img).astype(np.float32) / 255.0  # normalize 0-1
        arr = torch.from_numpy(arr).permute(2, 0, 1)     # [3,32,32]
        imgs.append(arr)

    return torch.stack(imgs)   # [num_samples,3,32,32]


# ---------- Worker for each GPU ----------
def ddpm_worker(rank, world_size, args):
    """
    rank: GPU index (0,1,2,...)
    world_size: total number of GPUs
    """
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load pipeline on this GPU
    print(f"[Rank {rank}] Loading pipeline on {device}...")
    pipe = DDPMPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.float16,
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    num_samples = args.num_samples
    batch_size = args.batch_size
    start_index = args.start_index

    # Decide which global indices this rank is responsible for
    # Global index range is [start_index, start_index + num_samples)
    # We partition this range evenly across GPUs.
    global_start_total = start_index
    global_end_total = start_index + num_samples

    start = global_start_total + (num_samples * rank // world_size)
    end = global_start_total + (num_samples * (rank + 1) // world_size)
    local_num = end - start

    if local_num <= 0:
        print(f"[Rank {rank}] No work assigned (local_num <= 0).")
        return

    print(
        f"[Rank {rank}] Generating indices [{start}, {end}) "
        f"({local_num} images) on {device}."
    )

    # Per-rank generator; shift seed by starting global index to avoid overlap
    generator = torch.Generator(device=device).manual_seed(args.seed + start)

    remaining = local_num
    global_idx = start

    # Simple progress bar per rank
    pbar = tqdm(
        total=local_num,
        desc=f"Rank {rank} DDPM",
        position=rank,
        leave=True,
    )

    while remaining > 0:
        cur_bs = min(batch_size, remaining)

        with torch.no_grad():
            out = pipe(batch_size=cur_bs, generator=generator)

        # Save each image with its global index
        for img in out.images:
            img_path = out_dir / f"ddpm_cifar10_{global_idx:05d}.png"
            img.save(img_path)
            global_idx += 1
            remaining -= 1
            pbar.update(1)

    pbar.close()
    print(f"[Rank {rank}] Done.")


# ---------- Main ----------
def main():
    parser = argparse.ArgumentParser(
        description="Multi-GPU DDPM sampling for CIFAR-10."
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        required=True,
        help="Directory to save PNGs (e.g., outputs/ddpm_cifar10)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50_000,
        help="Total number of samples to generate (across all GPUs).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Per-GPU batch size for pipeline(batch_size=...).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base random seed.",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/ddpm-cifar10-32",
        help="HuggingFace model id for DDPMPipeline.",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help=(
            "Global start index for naming output images. "
            "Images will be saved as ddpm_cifar10_{start-index + k:05d}.png."
        ),
    )
    parser.add_argument(
        "--skip-if-complete",
        action="store_true",
        help=(
            "If set, exits immediately when the full range "
            "[start-index, start-index + num-samples) already exists."
        ),
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Optional: if cache already complete, just exit
    if args.skip_if_complete:
        cached = load_ddpm_cache(out_dir, args.num_samples, start_index=args.start_index)
        if cached is not None:
            print(
                f"Cache already has images for indices "
                f"[{args.start_index}, {args.start_index + args.num_samples}) "
                f"in {out_dir}. Nothing to do."
            )
            return

    world_size = torch.cuda.device_count()
    if world_size <= 0:
        raise RuntimeError("No CUDA devices found.")
    print(f"Detected {world_size} GPUs.")

    # For safety with spawn
    mp.set_start_method("spawn", force=True)

    if world_size == 1:
        print("Using a single GPU.")
        ddpm_worker(rank=0, world_size=1, args=args)
    else:
        print(f"Using {world_size} GPUs with torch.multiprocessing.spawn.")
        mp.spawn(
            ddpm_worker,
            nprocs=world_size,
            args=(world_size, args),
        )

    print("All workers completed.")

    # Optional: sanity check after generation
    files = sorted(out_dir.glob("ddpm_cifar10_*.png"))
    print(f"Found {len(files)} PNG files in {out_dir}.")


if __name__ == "__main__":
    main()

