#!/usr/bin/env python3
"""Rewrite pyannote-segmentation-3.0 ONNX so the pure-Rust `tract` engine
can load it.

tract 0.22 fails to optimize the `InstanceNormalization` nodes in the
SincNet front-end ("Failed analyse ... InstanceNorm"). onnxruntime (the
sherpa export's target) supports the op natively, so the upstream file
keeps it. This script replaces every InstanceNormalization with its
mathematically-identical decomposition using ops tract does support
(ReduceMean/Sub/Mul/Add/Sqrt/Div), preserving output tensor names so the
rest of the graph is untouched.

Usage:
  python3 scripts/patch-pyannote-segmenter.py IN.onnx OUT.onnx

Verifies numerical equivalence against onnxruntime before writing.
"""
import sys
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto


def decompose_instance_norm(node, idx, inits):
    """Return (new_nodes, new_inits) replacing an InstanceNormalization.

    InstanceNorm over [N, C, L], normalizing the spatial axis (2):
        y = scale * (x - mean) / sqrt(var + eps) + bias
    """
    x, scale_name, bias_name = node.input[0], node.input[1], node.input[2]
    out = node.output[0]
    eps = 1e-5
    for a in node.attribute:
        if a.name == "epsilon":
            eps = float(helper.get_attribute_value(a))

    scale = numpy_helper.to_array(inits[scale_name])
    bias = numpy_helper.to_array(inits[bias_name])
    c = scale.shape[0]
    p = f"instnorm{idx}"

    scale_r = numpy_helper.from_array(
        scale.reshape(1, c, 1).astype(np.float32), f"{p}_scale"
    )
    bias_r = numpy_helper.from_array(
        bias.reshape(1, c, 1).astype(np.float32), f"{p}_bias"
    )
    eps_t = numpy_helper.from_array(np.array(eps, dtype=np.float32), f"{p}_eps")

    n = [
        helper.make_node("ReduceMean", [x], [f"{p}_mean"], axes=[2], keepdims=1),
        helper.make_node("Sub", [x, f"{p}_mean"], [f"{p}_xc"]),
        helper.make_node("Mul", [f"{p}_xc", f"{p}_xc"], [f"{p}_sq"]),
        helper.make_node("ReduceMean", [f"{p}_sq"], [f"{p}_var"], axes=[2], keepdims=1),
        helper.make_node("Add", [f"{p}_var", f"{p}_eps"], [f"{p}_vareps"]),
        helper.make_node("Sqrt", [f"{p}_vareps"], [f"{p}_std"]),
        helper.make_node("Div", [f"{p}_xc", f"{p}_std"], [f"{p}_norm"]),
        helper.make_node("Mul", [f"{p}_norm", f"{p}_scale"], [f"{p}_scaled"]),
        helper.make_node("Add", [f"{p}_scaled", f"{p}_bias"], [out]),
    ]
    return n, [scale_r, bias_r, eps_t]


def fold_if(nodes):
    """Replace no-op `If` nodes (identical then/else branches, e.g. the
    static-padding guard in SincNet) with the branch body inlined.

    tract can't execute `If` with a runtime condition; here both branches
    are byte-identical, so the condition is irrelevant and we splice one
    branch's nodes into the main graph, mapping branch outputs to the
    If's outputs. Returns (new_nodes, folded_count).
    """
    out, folded = [], 0
    for node in nodes:
        if node.op_type != "If":
            out.append(node)
            continue
        branches = {a.name: a.g for a in node.attribute if a.name in ("then_branch", "else_branch")}
        then_g, else_g = branches["then_branch"], branches["else_branch"]

        def sig(gr):
            return [(n.op_type, tuple(n.input), tuple(n.output)) for n in gr.node]

        assert sig(then_g) == sig(else_g), (
            "If branches differ — cannot fold blindly; needs condition eval"
        )
        # Map branch subgraph outputs -> the If node's outputs, positionally.
        remap = {bo.name: io for bo, io in zip(then_g.output, node.output)}
        for bn in then_g.node:
            new_out = [remap.get(o, o) for o in bn.output]
            del bn.output[:]
            bn.output.extend(new_out)
            out.append(bn)
        folded += 1
    return out, folded


def eliminate_dead_nodes(g):
    """Iteratively drop nodes whose outputs feed nothing (and are not graph
    outputs), plus orphaned initializers. Removes the now-dead If-condition
    chain (Shape/Gather/Equal/Cast) and the InstanceNorm scale/bias that
    the decomposition replaced."""
    graph_outputs = {o.name for o in g.output}
    removed = 0
    while True:
        used = set(graph_outputs)
        for n in g.node:
            used.update(n.input)
        drop = [
            n
            for n in g.node
            if n.output and not any(o in used for o in n.output)
        ]
        if not drop:
            break
        for n in drop:
            g.node.remove(n)
        removed += len(drop)

    used = set(graph_outputs)
    for n in g.node:
        used.update(n.input)
    keep = [i for i in g.initializer if i.name in used]
    dropped_inits = len(g.initializer) - len(keep)
    del g.initializer[:]
    g.initializer.extend(keep)
    return removed, dropped_inits


def main():
    src, dst = sys.argv[1], sys.argv[2]
    m = onnx.load(src)
    g = m.graph
    inits = {i.name: i for i in g.initializer}

    new_nodes, added_inits, replaced = [], [], 0
    for i, node in enumerate(g.node):
        if node.op_type == "InstanceNormalization":
            nodes, extra = decompose_instance_norm(node, i, inits)
            new_nodes.extend(nodes)
            added_inits.extend(extra)
            replaced += 1
        else:
            new_nodes.append(node)

    new_nodes, folded = fold_if(new_nodes)

    del g.node[:]
    g.node.extend(new_nodes)
    g.initializer.extend(added_inits)

    dead, dropped_inits = eliminate_dead_nodes(g)
    print(
        f"replaced {replaced} InstanceNorm, folded {folded} If, "
        f"removed {dead} dead node(s) + {dropped_inits} orphan initializer(s)"
    )

    # Bake the fixed [1, 1, 80000] input shape and re-run shape inference so
    # every intermediate value_info is concrete — helps tract's analysis
    # pass resolve broadcasts without ambiguity.
    inp = g.input[0]
    dims = inp.type.tensor_type.shape.dim
    for d, v in zip(dims, (1, 1, 80_000)):
        d.ClearField("dim_param")
        d.dim_value = v
    for vi in list(g.value_info):
        g.value_info.remove(vi)
    m = onnx.shape_inference.infer_shapes(m)

    onnx.checker.check_model(m)
    onnx.save(m, dst)
    print(f"wrote {dst}")

    # Numerical equivalence check vs the original.
    import onnxruntime as ort

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 1, 80_000)).astype(np.float32)
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    a = ort.InferenceSession(src, opts, providers=["CPUExecutionProvider"])
    b = ort.InferenceSession(dst, opts, providers=["CPUExecutionProvider"])
    ya = a.run(None, {"x": x})[0]
    yb = b.run(None, {"x": x})[0]
    max_abs = float(np.max(np.abs(ya - yb)))
    print(f"output shapes: {ya.shape} vs {yb.shape}")
    print(f"max abs diff vs original: {max_abs:.3e}")
    assert ya.shape == yb.shape, "shape mismatch after patch"
    assert max_abs < 1e-4, f"numerical drift too large: {max_abs}"
    print("EQUIVALENCE OK")


if __name__ == "__main__":
    main()
