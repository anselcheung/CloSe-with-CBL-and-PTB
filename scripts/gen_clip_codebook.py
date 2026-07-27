"""Generate a CLIP-derived semantic prior for CloSeNet's garment codebook.

Dependency: OpenAI CLIP (`pip install git+https://github.com/openai/CLIP.git`). CLIP
has no version tags, so for reproducibility, record the exact commit hash of the
installed package (`pip show clip` / `python -c "import clip; print(clip.__file__)"`
plus `git -C <clip checkout> rev-parse HEAD` if installed editable) alongside the
generated `.npy`. Do NOT substitute SigLIP, MetaCLIP, or any other CLIP variant.

Usage (this repo's codebook dim is 64, cfg.model_arch.garm_enc.gar_emb_dim in
cfg/closenet.yaml):
    python scripts/gen_clip_codebook.py --l 64
"""
import argparse
from pathlib import Path

import clip
import numpy as np
import torch

# ORDER MUST MATCH docs/dataset.md label mapping — verify before running.
CLASS_NAMES = ["Hat", "Body", "Shirt", "TShirt", "Vest", "Coat", "Dress", "Skirt",
               "Pants", "ShortPants", "Shoes", "Hoodies", "Hair", "Swimwear",
               "Underwear", "Scarf", "Jumpsuits", "Jacket"]


def main(l: int, out_path: str, seed: int = 0) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = clip.load("ViT-B/32", device=device)
    prompts = [f"a photo of {name.lower()}" for name in CLASS_NAMES]
    tokens = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        emb = model.encode_text(tokens).float().cpu().numpy()  # (18, 512)

    # Fixed random projection 512 -> l (JL-style; deterministic given seed). PCA is not
    # used: with only 18 samples it has at most 17 nonzero components.
    rng = np.random.default_rng(seed)
    proj = rng.standard_normal((512, l)) / np.sqrt(512)
    codebook_init = (emb @ proj).astype(np.float32)  # (18, l)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, codebook_init)
    print(f"Wrote {codebook_init.shape} to {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--l", type=int, required=True, help="codebook dim (must match model_arch.garm_enc.gar_emb_dim)")
    ap.add_argument("--out", type=str, default="data/clip_codebook_init.npy")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    main(l=args.l, out_path=args.out, seed=args.seed)
