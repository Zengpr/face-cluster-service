"""Generate synthetic but visually distinct face-like images for smoke tests.

These are NOT real faces — they are simple placeholder gradients/scribbles.
They let us verify the full HTTP + Docker + clustering pipeline without
shipping real biometric data in a public GitHub repo.

A real-face test path is described in docs/TESTING.md.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _seed_from(name: str) -> int:
    return int(hashlib.md5(name.encode()).hexdigest()[:8], 16)


def _identity_color(seed: int) -> tuple[int, int, int]:
    rng = np.random.default_rng(seed)
    return tuple(int(v) for v in (rng.integers(40, 220, size=3)))


def _synthetic_face(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (size, size), color=_identity_color(seed))
    draw = ImageDraw.Draw(img)

    base = seed
    # Head ellipse
    draw.ellipse(
        [size * 0.18, size * 0.10, size * 0.82, size * 0.80],
        outline=tuple(int(c * 0.6) for c in _identity_color(seed + 1)),
        width=int(size * 0.02),
    )
    # Eyes — perturbed deterministically per identity so similar
    # identity seeds yield similar eye placement.
    eye_y = int(size * (0.35 + 0.05 * rng.standard_normal()))
    eye_dx = int(size * (0.10 + 0.04 * rng.standard_normal()))
    eye_radius = int(size * (0.04 + 0.015 * rng.standard_normal()))
    draw.ellipse(
        [size // 2 - eye_dx - eye_radius, eye_y - eye_radius,
         size // 2 - eye_dx + eye_radius, eye_y + eye_radius],
        fill=(20, 20, 20),
    )
    draw.ellipse(
        [size // 2 + eye_dx - eye_radius, eye_y - eye_radius,
         size // 2 + eye_dx + eye_radius, eye_y + eye_radius],
        fill=(20, 20, 20),
    )
    # Mouth
    mouth_y = int(size * 0.60)
    draw.line(
        [(size // 2 - 30, mouth_y), (size // 2 + 30, mouth_y)],
        fill=(40, 20, 20), width=int(size * 0.02),
    )
    return img


def main(out_dir: Path, per_identity: int = 3, n_identities: int = 3) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    truth: dict[str, list[str]] = {}
    for ident_idx in range(n_identities):
        ident_seed = ident_idx + 1000
        files: list[str] = []
        for shot_idx in range(per_identity):
            shot_seed = ident_seed * 17 + shot_idx * 7
            img = _synthetic_face(shot_seed)
            # Add identity-locked small noise so two photos of same
            # "person" are similar in colour & shape (this won't be
            # detected as a real face by buffalo_l — that's OK for
            # this smoke test, see docs/TESTING.md for real-face run).
            arr = np.array(img)
            noise = (np.random.default_rng(ident_seed).standard_normal(arr.shape) * 4)
            arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
            fname = f"ident{ident_idx}_shot{shot_idx}.png"
            Image.fromarray(arr).save(out_dir / fname, format="PNG")
            files.append(fname)
        truth[f"cluster_{ident_idx}"] = files
    return truth


if __name__ == "__main__":
    import json
    import sys

    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/data/images")
    truth = main(root)
    print(f"wrote {sum(len(v) for v in truth.values())} images to {root}")
    print(json.dumps(truth, indent=2))
