"""Benchmark sequential vs batched CFG transformer forwards.

Compares per-step CFG timing and verifies that batched mode produces
bit-identical latents relative to the original sequential path.
"""

from __future__ import annotations

import argparse
import sys

import torch

from ideogram4 import Ideogram4Pipeline, Ideogram4PipelineConfig


def _default_device() -> str:
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def _default_quantization() -> str:
  return "nf4" if torch.cuda.is_available() else "fp8"


QUANTIZATION_REPOS = {
  "nf4": "ideogram-ai/ideogram-4-nf4",
  "fp8": "ideogram-ai/ideogram-4-fp8",
}

# Minimal valid Ideogram 4 JSON caption for benchmarking without magic-prompt.
BENCHMARK_PROMPT = """{
  "high_level_description": "A red apple on a wooden table.",
  "style_description": {
    "aesthetics": "photorealistic product photography",
    "lighting": "soft natural daylight",
    "photo": "centered composition, shallow depth of field",
    "medium": "photograph",
    "color_palette": ["#CC0000", "#8B4513", "#FFFFFF"]
  },
  "compositional_deconstruction": {
    "background": "A warm wooden table surface with soft natural light.",
    "elements": [
      {
        "type": "obj",
        "bbox": [300, 250, 700, 750],
        "desc": "A glossy red apple with a short brown stem, centered on the table."
      }
    ]
  }
}"""


def main() -> None:
  parser = argparse.ArgumentParser(
    description="Benchmark batched vs sequential CFG transformer forwards."
  )
  parser.add_argument("--prompt", default=BENCHMARK_PROMPT)
  parser.add_argument("--height", type=int, default=512)
  parser.add_argument("--width", type=int, default=512)
  parser.add_argument("--num-steps", type=int, default=20)
  parser.add_argument("--warmup", type=int, default=3)
  parser.add_argument("--guidance-scale", type=float, default=7.0)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default=_default_device())
  parser.add_argument(
    "--quantization",
    choices=sorted(QUANTIZATION_REPOS.keys()),
    default=_default_quantization(),
  )
  parser.add_argument(
    "--generate-images",
    action="store_true",
    help="Also decode latents to PNGs for visual comparison.",
  )
  args = parser.parse_args()

  if args.device == "cuda" and not torch.cuda.is_available():
    print("ERROR: --device cuda requested but CUDA is not available.", file=sys.stderr)
    sys.exit(1)

  print(
    f"Loading pipeline ({args.quantization}, device={args.device})...",
    file=sys.stderr,
  )
  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(weights_repo=QUANTIZATION_REPOS[args.quantization]),
    device=args.device,
    dtype=torch.bfloat16,
  )

  print(
    f"Running CFG benchmark: {args.num_steps} steps, "
    f"{args.warmup} warmup, {args.height}x{args.width}",
    file=sys.stderr,
  )
  result = pipe.benchmark_cfg(
    args.prompt,
    height=args.height,
    width=args.width,
    num_steps=args.num_steps,
    num_warmup=args.warmup,
    guidance_scale=args.guidance_scale,
    seed=args.seed,
  )

  print("\n=== CFG benchmark results ===")
  print(f"Device:              {args.device}")
  print(f"Resolution:          {args.height}x{args.width}")
  print(f"Steps measured:      {result.num_steps}")
  print(f"Warmup iterations:   {result.num_warmup}")
  print(f"Sequential CFG:      {result.separate_ms_per_step:.2f} ms/step")
  print(f"Batched CFG:         {result.batched_ms_per_step:.2f} ms/step")
  print(f"Speedup:             {result.speedup:.2f}x")
  print(f"Max latent |diff|:    {result.max_latent_diff:.3e}")

  if result.max_latent_diff == 0.0:
    print("Numerical equivalence: PASS (bit-identical latents)")
  elif result.max_latent_diff < 1e-5:
    print("Numerical equivalence: PASS (within float32 tolerance)")
  else:
    print("Numerical equivalence: FAIL")
    sys.exit(1)

  if args.device != "cuda":
    print(
      "\nNote: batched CFG overlaps transformer forwards on separate CUDA streams. "
      f"On {args.device}, batched mode falls back to sequential execution."
    )

  if args.generate_images:
    print("\nGenerating comparison images...", file=sys.stderr)
    for label, use_batched in (("sequential", False), ("batched", True)):
      images = pipe(
        args.prompt,
        height=args.height,
        width=args.width,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        raise_on_caption_issues=False,
        use_batched_cfg=use_batched,
      )
      out_path = f"benchmark_cfg_{label}.png"
      images[0].save(out_path)
      print(f"Saved {out_path}", file=sys.stderr)


if __name__ == "__main__":
  main()
