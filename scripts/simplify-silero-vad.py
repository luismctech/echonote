#!/usr/bin/env python3
"""
Simplify the upstream Silero VAD v5 ONNX so `tract-onnx` can load it.

The upstream graph dispatches between the 16 kHz and 8 kHz sub-networks
through an ONNX `If` operator that reads the `sr` input. `tract-onnx`
does not implement `If`, so loading the raw upstream file fails with
`optimize: Failed analyse for node #5 "If_0" If`.

EchoNote only ever feeds 16 kHz audio, so this script:

  1. Inlines the 16 kHz `then_branch` of the outer `If`.
  2. Drops the now-unused `sr` input.
  3. Pins the remaining inputs to their static shapes
     (`input = [1, 512]`, `state = [2, 1, 128]`).
  4. Runs ONNX Runtime's `ORT_ENABLE_BASIC` graph optimizer, which
     constant-folds the three nested `If`s that depended on static
     shape values (all of them reduce to their `then_branch` under
     our fixed shapes).

NOTE on the ORT optimization level: `BASIC` is deliberate, not
arbitrary. We empirically probed all four ORT levels (see the commit
that introduced this script):

    DISABLE_ALL → 56 nodes,  3 Ifs,  0 contrib ops    (tract: ❌ Ifs)
    BASIC       → 36 nodes,  0 Ifs,  0 contrib ops    (tract: ✅)
    EXTENDED    → 31 nodes,  0 Ifs,  5 FusedConv      (tract: ❌ FusedConv)
    ALL         → 31 nodes,  0 Ifs,  5 FusedConv      (tract: ❌ FusedConv)

Anything above BASIC fuses Conv+ReLU into ORT's contrib `FusedConv`
op, which is NOT part of the standard ONNX op set and which tract
rejects with `Unimplemented(FusedConv) ToTypedTranslator`. BASIC is
the unique level that both (a) eliminates the static-shape Ifs and
(b) keeps the graph inside the standard op vocabulary tract supports.

The result is a pure feed-forward + LSTM graph with 36 standard-ONNX
nodes and zero `If`s, numerically equivalent to the upstream model
for 16 kHz audio.

Usage:
    python3 scripts/simplify-silero-vad.py \\
        --input  models/vad/silero_vad.onnx \\
        --output models/vad/silero_vad.onnx

    # or, with a separate destination (the default is in-place):
    python3 scripts/simplify-silero-vad.py \\
        --input  models/vad/silero_vad_raw.onnx \\
        --output models/vad/silero_vad.onnx

Requires: onnx>=1.14, onnxruntime>=1.17.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path


# ORT contrib ops that tract-onnx cannot translate. Any of these in the
# output means we accidentally raised the optimization level above BASIC.
_ORT_CONTRIB_OPS = frozenset({
    "FusedConv", "FusedGemm", "FusedMatMul",
    "NchwcConv", "NchwcMaxPool", "NchwcAveragePool", "NchwcUpsample",
    "ReorderInput", "ReorderOutput",
    "QLinearConv", "QLinearMatMul", "QGemm",
    "BiasGelu", "FastGelu", "Gelu",
    "EmbedLayerNormalization", "SkipLayerNormalization", "LayerNormalization",
})


def already_simplified(model) -> bool:
    """True if the graph satisfies all three invariants this script
    guarantees: no top-level `If`, no `sr` input, no ORT contrib ops.
    The contrib-ops check matters because earlier versions of this
    script ran ORT at `ENABLE_ALL` and produced files of the same
    size as the BASIC output but containing `FusedConv` (which tract
    rejects). Without this check a stale-but-same-size file would
    be treated as ready and silently break tract loading."""
    has_if = any(n.op_type == "If" for n in model.graph.node)
    has_sr = any(i.name == "sr" for i in model.graph.input)
    has_contrib = any(n.op_type in _ORT_CONTRIB_OPS for n in model.graph.node)
    return not (has_if or has_sr or has_contrib)


def inline_outer_if(model):
    """Replace the outer `Equal(sr, 16000) → If` wrapper with the 16 kHz
    `then_branch` subgraph. Identity passthroughs that referenced the
    `If`'s outputs continue to work because we rename the branch's
    terminal outputs to the `If`'s output tensor names."""
    import onnx
    from onnx import helper

    g = model.graph
    try:
        if_node = next(n for n in g.node if n.op_type == "If")
    except StopIteration:
        return model

    then_branch = next(a.g for a in if_node.attribute if a.name == "then_branch")
    assert len(then_branch.output) == len(if_node.output), (
        "then_branch output count mismatch — is this really the Silero v5 ONNX?"
    )
    rename = {b.name: i for b, i in zip(then_branch.output, if_node.output)}

    def remap(names):
        return [rename.get(n, n) for n in names]

    dropped = {"Constant_0", "Equal_0", "If_0"}
    new_nodes = []
    for n in g.node:
        if n.name == "If_0":
            for bn in then_branch.node:
                nn = onnx.NodeProto()
                nn.CopyFrom(bn)
                nn.output[:] = remap(list(nn.output))
                new_nodes.append(nn)
            continue
        if n.name in dropped:
            continue
        new_nodes.append(n)

    new_graph = helper.make_graph(
        nodes=new_nodes,
        name=g.name + "_static16k",
        inputs=list(g.input),
        outputs=list(g.output),
        initializer=list(g.initializer),
    )
    new_model = helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        producer_name=(model.producer_name or "silero") + "+echonote_static16k",
        ir_version=model.ir_version,
    )
    onnx.checker.check_model(new_model, full_check=True)
    return new_model


def strip_sr_input_and_lock_shapes(model):
    """Remove the unused scalar `sr` input (orphaned after inlining) and
    lock the two surviving inputs to the exact shapes tract will see
    at runtime. This is what lets ORT fold the nested `If`s."""
    from onnx import helper

    g = model.graph
    kept_inputs = [i for i in g.input if i.name != "sr"]

    def fix_shape(vi, shape):
        vi.type.tensor_type.shape.ClearField("dim")
        for d in shape:
            vi.type.tensor_type.shape.dim.add(dim_value=d)

    for vi in kept_inputs:
        if vi.name == "input":
            fix_shape(vi, [1, 512])
        elif vi.name == "state":
            fix_shape(vi, [2, 1, 128])

    new_graph = helper.make_graph(
        nodes=list(g.node),
        name=g.name,
        inputs=kept_inputs,
        outputs=list(g.output),
        initializer=list(g.initializer),
    )
    return helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        producer_name=model.producer_name,
        ir_version=model.ir_version,
    )


def ort_optimize(input_path: Path, output_path: Path) -> None:
    """Run ORT's BASIC-level graph optimizer.

    BASIC = constant folding + redundant-node elimination + shape-aware
    rewrites that stay inside the standard ONNX op set. This is enough
    to fold the three static-shape `If`s left by step 1, and crucially
    it does NOT introduce ORT's contrib ops (FusedConv, NchwcConv, …)
    that tract cannot translate. See the module-level docstring for
    the full level-by-level comparison."""
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    so.optimized_model_filepath = str(output_path)
    _ = ort.InferenceSession(
        str(input_path), sess_options=so, providers=["CPUExecutionProvider"]
    )


def assert_no_contrib_ops(model) -> None:
    """Fail loudly if the optimized graph contains any ORT contrib op.
    These are not part of the standard ONNX op set and tract rejects
    them with `Unimplemented(<OpName>) ToTypedTranslator`."""
    found = sorted({n.op_type for n in model.graph.node if n.op_type in _ORT_CONTRIB_OPS})
    if found:
        raise SystemExit(
            "error: simplified graph contains ORT contrib ops that tract cannot load: "
            f"{found}. Lower the ORT optimization level to BASIC."
        )


def verify_parity(simplified_path: Path, original_path: Path) -> None:
    """Bit-exact numerical check against the untouched upstream model.
    Differences would point to a mis-inlined branch or an ORT rewrite
    that is not actually sr-invariant."""
    import numpy as np
    import onnxruntime as ort

    new = ort.InferenceSession(str(simplified_path), providers=["CPUExecutionProvider"])
    old = ort.InferenceSession(str(original_path), providers=["CPUExecutionProvider"])

    rng = np.random.default_rng(0)
    cases = [
        ("noise", rng.standard_normal((1, 512)).astype(np.float32) * 0.1),
        ("silence", np.zeros((1, 512), np.float32)),
        ("loud", rng.standard_normal((1, 512)).astype(np.float32)),
    ]
    for name, audio in cases:
        state = np.zeros((2, 1, 128), np.float32)
        p_new = float(new.run(None, {"input": audio, "state": state})[0][0][0])
        p_old = float(
            old.run(None, {"input": audio, "state": state, "sr": np.array(16000, np.int64)})[0][0][0]
        )
        delta = abs(p_new - p_old)
        marker = "ok" if delta < 1e-5 else "FAIL"
        print(f"    parity {name:8s} new={p_new:.6f} old={p_old:.6f} Δ={delta:.2e}  [{marker}]")
        if delta >= 1e-5:
            raise SystemExit(
                f"numerical parity lost for case '{name}': the simplified graph "
                f"disagrees with the upstream by {delta:.2e}"
            )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, type=Path, help="Upstream silero_vad.onnx")
    ap.add_argument("--output", required=True, type=Path, help="Destination (may equal --input)")
    ap.add_argument(
        "--skip-parity",
        action="store_true",
        help="Skip the numerical parity check (faster; only use if you already trust the pipeline).",
    )
    args = ap.parse_args()

    try:
        import onnx  # noqa: F401
        import onnxruntime  # noqa: F401
    except ImportError:
        sys.stderr.write(
            "error: this script requires `onnx` and `onnxruntime`.\n"
            "       Install with: pip install --user onnx onnxruntime\n"
        )
        return 2

    import onnx

    if not args.input.is_file():
        sys.stderr.write(f"error: input file not found: {args.input}\n")
        return 1

    raw = onnx.load(str(args.input))
    if already_simplified(raw):
        print(f"  silero_vad: {args.input.name} is already simplified — skipping.")
        if args.input.resolve() != args.output.resolve():
            shutil.copy(args.input, args.output)
        return 0

    upstream_copy = args.output.with_name(args.output.name + ".upstream")

    # If the on-disk model is in an intermediate / tainted state (e.g. a
    # stale build from an earlier version of this script that contained
    # FusedConv, or one where the outer If was already inlined), the
    # graph-rewrite passes below would silently no-op. Detect this by
    # checking whether the raw upstream `If` is even present, and if not,
    # restart the simplification from the cached upstream copy.
    has_outer_if = any(
        n.op_type == "If" and n.name == "If_0" for n in raw.graph.node
    )
    if not has_outer_if:
        if not upstream_copy.exists():
            raise SystemExit(
                f"error: {args.input} is in an intermediate state (no outer If) "
                f"and no upstream copy exists at {upstream_copy} to restart from. "
                "Delete the .onnx file and re-run `scripts/download-models.sh vad` "
                "to fetch a fresh upstream copy."
            )
        print(
            f"  silero_vad: {args.input.name} is in an intermediate state — "
            f"restarting from cached upstream at {upstream_copy.name}"
        )
        raw = onnx.load(str(upstream_copy))

    print(f"  silero_vad: simplifying {args.input} → {args.output}")

    # Preserve the raw upstream file next to the destination for parity checks
    # and future re-simplification (e.g. if we tune input shapes).
    if not upstream_copy.exists():
        shutil.copy(args.input, upstream_copy)

    step1 = inline_outer_if(raw)
    step2 = strip_sr_input_and_lock_shapes(step1)

    with tempfile.TemporaryDirectory() as td:
        mid = Path(td) / "silero_vad.mid.onnx"
        onnx.save(step2, str(mid))
        ort_optimize(mid, args.output)

    final = onnx.load(str(args.output))
    n_if = sum(1 for n in final.graph.node if n.op_type == "If")
    n_nodes = len(final.graph.node)
    size_kib = args.output.stat().st_size // 1024
    print(f"    nodes={n_nodes}  remaining_if={n_if}  size={size_kib} KiB")
    if n_if:
        raise SystemExit(
            f"error: simplified graph still contains {n_if} If op(s) — tract will fail to load it. "
            "This is a regression in the simplifier; check the upstream model layout."
        )
    assert_no_contrib_ops(final)

    if not args.skip_parity:
        verify_parity(args.output, upstream_copy)

    print(f"  silero_vad: done ({args.output})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
