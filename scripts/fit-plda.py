#!/usr/bin/env python3
"""Fit the LDA + two-covariance PLDA back-end of the VBx re-segmentation.

The Rust reader is `crates/echo-diarize/src/plda.rs`; its module docs define
the model and the file format. The recipe follows BUTSpeechFIT VBx
(`vbhmm.py`, Apache-2.0): embeddings are centred with `mean1`,
length-normalised, projected with an LDA to `lda_dim` dimensions, centred
with `mean2` and length-normalised again; a two-covariance PLDA is then
fitted on that space and diagonalised (`eigh(B, W)`), so the within-speaker
covariance is the identity and the across-speaker one is `diag(psi)`.

Input: one or more embedding dumps written by
`echo-proto bench embed-dump` (magic `ECHOEMB1`, `u32` rows, `u32` dim,
row-major little-endian f32) with their `<dump>.labels.txt` sidecars (one
speaker label per row). Speakers with fewer than `--min-per-speaker`
embeddings are dropped.

Usage (in an isolated venv with numpy and scipy, run with `python -I`):

    python -I scripts/fit-plda.py --lda-dim 128 --out plda.bin dump1.bin [dump2.bin ...]

The training data must be speaker-disjoint from every evaluation set, and
its licence must allow distributing a model derived from it.
"""

import argparse
import hashlib
import struct
import sys

import numpy as np
from scipy.linalg import eigh

MAGIC_IN = b"ECHOEMB1"
MAGIC_OUT = b"ECHOPLDA"
VERSION = 1


def read_dump(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:8] != MAGIC_IN:
        sys.exit(f"{path}: not an embedding dump")
    rows, dim = struct.unpack("<II", data[8:16])
    x = np.frombuffer(data, dtype="<f4", offset=16)
    if x.size != rows * dim:
        sys.exit(f"{path}: expected {rows}x{dim} values, found {x.size}")
    with open(f"{path}.labels.txt", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if len(labels) != rows:
        sys.exit(f"{path}: {rows} rows but {len(labels)} labels")
    return x.reshape(rows, dim).astype(np.float64), labels


def l2_norm(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def class_stats(x, y):
    """Global mean, within-class and between-class covariances, class count.

    The between-class covariance is that of the class means, corrected for
    the within-class noise each mean still carries (W / mean class size).
    """
    mean = x.mean(axis=0)
    classes = np.unique(y)
    d = x.shape[1]
    within = np.zeros((d, d))
    means = []
    sizes = []
    for c in classes:
        xc = x[y == c]
        mc = xc.mean(axis=0)
        within += (xc - mc).T @ (xc - mc)
        means.append(mc)
        sizes.append(len(xc))
    within /= len(x)
    means = np.array(means) - mean
    between = means.T @ means / len(classes)
    between -= within / np.mean(sizes)
    # Keep the across-class covariance positive semi-definite.
    w, v = np.linalg.eigh((between + between.T) / 2)
    between = (v * np.maximum(w, 1e-6 * w.max())) @ v.T
    return mean, within, between, len(classes)


def fit(x, labels, lda_dim, min_per_speaker):
    labels = np.array(labels)
    keep_spk = [s for s in np.unique(labels) if (labels == s).sum() >= min_per_speaker]
    mask = np.isin(labels, keep_spk)
    x, y = x[mask], labels[mask]
    if len(keep_spk) <= lda_dim // 4:
        print(f"warning: only {len(keep_spk)} speakers for lda_dim {lda_dim}", file=sys.stderr)

    mean1 = x.mean(axis=0)
    x1 = l2_norm(x - mean1)
    _, sw, sb, n_spk = class_stats(x1, y)
    # LDA: generalised eigenvectors of (Sb, Sw), largest first; eigh
    # normalises them so the projected within-class covariance is I.
    _, vecs = eigh(sb, sw + 1e-6 * np.eye(sw.shape[0]) * np.trace(sw) / sw.shape[0])
    lda = vecs[:, ::-1][:, :lda_dim]
    proj = x1 @ lda
    mean2 = proj.mean(axis=0)
    z = l2_norm(proj - mean2)

    mu, w, b, _ = class_stats(z, y)
    acvar, wccn = eigh(b, w)
    psi = np.maximum(acvar[::-1], 0.0)
    tr = wccn.T[::-1]
    return {
        "mean1": mean1,
        "lda": lda,
        "mean2": mean2,
        "mu": mu,
        "tr": tr,
        "psi": psi,
        "speakers": n_spk,
        "rows": len(x),
    }


def encode(model):
    dim, lda_dim = model["lda"].shape
    out = bytearray(MAGIC_OUT)
    out += struct.pack("<III", VERSION, dim, lda_dim)
    for key in ("mean1", "lda", "mean2", "mu", "tr", "psi"):
        a = np.ascontiguousarray(model[key], dtype="<f4")
        if not np.all(np.isfinite(a)):
            sys.exit(f"non-finite values in {key}")
        out += a.tobytes()
    return bytes(out)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("dumps", nargs="+", help="embedding dumps from `bench embed-dump`")
    p.add_argument("--lda-dim", type=int, default=128)
    p.add_argument("--min-per-speaker", type=int, default=8)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    xs, labels = [], []
    for path in args.dumps:
        x, lab = read_dump(path)
        xs.append(x)
        labels += lab
    x = np.concatenate(xs)
    if args.lda_dim > x.shape[1]:
        sys.exit(f"--lda-dim {args.lda_dim} exceeds the embedding dimension {x.shape[1]}")
    model = fit(x, labels, args.lda_dim, args.min_per_speaker)
    data = encode(model)
    with open(args.out, "wb") as f:
        f.write(data)
    print(
        f"{model['rows']} embeddings, {model['speakers']} speakers, "
        f"dim {x.shape[1]} -> {args.lda_dim}; psi[:5] {np.round(model['psi'][:5], 3).tolist()}; "
        f"{len(data)} bytes, sha256 {hashlib.sha256(data).hexdigest()}"
    )


if __name__ == "__main__":
    main()
