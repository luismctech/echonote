#!/usr/bin/env python3
"""
Inline the 16 kHz branch of Silero VAD v6.2 so `tract-onnx` can load it.

Upstream ships `silero_vad_op18_ifless.onnx`, whose nested shape `If`s
are already folded away, but which still dispatches between the 16 kHz
and 8 kHz sub-networks through ONE top-level `If` controlled by the
`sr` input:

    Reshape(sr) -> Gather -> Equal(·, 16000) -> If -> (output, stateN)

`tract-onnx` does not implement `If`, so loading the raw file fails.
EchoNote only ever feeds 16 kHz audio, so this script:

  1. Confirms the `If` condition is `sr == 16000` and takes its
     `then_branch` (the 16 kHz network; the `else_branch` reads the
     `model_8k.*` weights).
  2. Splices that branch into the main graph, renaming its terminal
     outputs to the `If` outputs (`output`, `stateN`).
  3. Drops the dispatch nodes, the `sr` input and every initializer the
     inlined graph no longer references (the 8 kHz weights).

Input shapes stay symbolic: the Rust adapter pins `input = [1, 576]`
(64 context samples + 512-sample hop) and `state = [2, 1, 128]`.

Only `onnx` is needed; no ONNX Runtime pass is involved. The script is
idempotent: an input without `If` nodes and without an `sr` input is
treated as already inlined and copied through unchanged.

Usage:
    python3 scripts/inline-silero-vad.py \\
        --input  models/vad/silero_vad.onnx.upstream \\
        --output models/vad/silero_vad.onnx

Requires: onnx>=1.14.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

# The `If` condition compares `sr` against this constant; its
# `then_branch` is therefore the 16 kHz network.
_SR_16K = 16_000


def already_inlined(model) -> bool:
    """True if the graph has no `If` node and no `sr` input."""
    has_if = any(n.op_type == "If" for n in model.graph.node)
    has_sr = any(i.name == "sr" for i in model.graph.input)
    return not (has_if or has_sr)


def _constant_value(graph, name: str):
    """Value of `name` if it is a graph initializer or `Constant` output."""
    from onnx import numpy_helper

    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    for node in graph.node:
        if node.op_type == "Constant" and name in node.output:
            return numpy_helper.to_array(node.attribute[0].t)
    return None


def _check_condition_is_16k(graph, if_node) -> None:
    """Fail unless the `If` condition is `Equal(<sr-derived>, 16000)`.

    Guards against an upstream re-export that flips the comparison, in
    which case `then_branch` would silently be the 8 kHz network."""
    cond = if_node.input[0]
    equal = next((n for n in graph.node if cond in n.output), None)
    if equal is None or equal.op_type != "Equal":
        raise SystemExit(f"error: If condition `{cond}` is not produced by an Equal node")
    values = [_constant_value(graph, name) for name in equal.input]
    if not any(v is not None and v.size == 1 and int(v.item()) == _SR_16K for v in values):
        raise SystemExit(
            f"error: If condition does not compare sr against {_SR_16K}; "
            "refusing to guess which branch is the 16 kHz network"
        )


def inline_16k_branch(model):
    """Return a copy of `model` with the 16 kHz `then_branch` inlined."""
    import onnx
    from onnx import helper

    g = model.graph
    if_nodes = [n for n in g.node if n.op_type == "If"]
    if len(if_nodes) != 1:
        raise SystemExit(
            f"error: expected exactly one top-level If, found {len(if_nodes)} — "
            "is this really silero_vad_op18_ifless.onnx?"
        )
    if_node = if_nodes[0]
    _check_condition_is_16k(g, if_node)

    then_branch = next(a.g for a in if_node.attribute if a.name == "then_branch")
    if len(then_branch.output) != len(if_node.output):
        raise SystemExit("error: then_branch output count does not match the If outputs")
    rename = {b.name: o for b, o in zip(then_branch.output, if_node.output)}

    nodes = []
    for bn in then_branch.node:
        nn = onnx.NodeProto()
        nn.CopyFrom(bn)
        nn.output[:] = [rename.get(o, o) for o in bn.output]
        nodes.append(nn)
    if any(n.op_type == "If" for n in nodes):
        raise SystemExit("error: the 16 kHz branch itself contains an If; tract cannot load it")

    # The dispatch nodes (Reshape/Gather/Equal on `sr`) are dropped by
    # construction: only branch nodes are kept. Keep just the
    # initializers they reference, which discards the 8 kHz weights.
    used = {name for n in nodes for name in n.input}
    initializers = [
        i for i in list(g.initializer) + list(then_branch.initializer) if i.name in used
    ]
    inputs = [i for i in g.input if i.name != "sr"]

    new_graph = helper.make_graph(nodes, g.name, inputs, list(g.output), initializers)
    new_model = helper.make_model(
        new_graph,
        opset_imports=list(model.opset_import),
        producer_name=(model.producer_name or "silero") + "+echonote_static16k",
        ir_version=model.ir_version,
    )
    onnx.checker.check_model(new_model, full_check=True)
    return new_model


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--input", required=True, type=Path, help="Upstream silero_vad_op18_ifless.onnx")
    ap.add_argument("--output", required=True, type=Path, help="Destination (may equal --input)")
    args = ap.parse_args()

    try:
        import onnx
    except ImportError:
        sys.stderr.write(
            "error: this script requires `onnx`.\n"
            "       Install with: pip install --user onnx\n"
        )
        return 2

    if not args.input.is_file():
        sys.stderr.write(f"error: input file not found: {args.input}\n")
        return 1

    raw = onnx.load(str(args.input))
    if already_inlined(raw):
        print(f"  silero_vad: {args.input.name} is already inlined — skipping.")
        if args.input.resolve() != args.output.resolve():
            shutil.copy(args.input, args.output)
        return 0

    print(f"  silero_vad: inlining 16 kHz branch {args.input} → {args.output}")
    model = inline_16k_branch(raw)
    onnx.save(model, str(args.output))

    names_in = [i.name for i in model.graph.input]
    names_out = [o.name for o in model.graph.output]
    size_kib = args.output.stat().st_size // 1024
    print(
        f"    nodes={len(model.graph.node)}  inputs={names_in}  "
        f"outputs={names_out}  size={size_kib} KiB"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
