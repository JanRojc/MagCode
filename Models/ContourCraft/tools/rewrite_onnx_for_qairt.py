import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, numpy_helper


def _value_dim(value, idx: int) -> int | None:
    dims = value.type.tensor_type.shape.dim
    if idx >= len(dims):
        return None
    dim = dims[idx]
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    return None


def _set_first_dim(value, rows: int) -> None:
    dims = value.type.tensor_type.shape.dim
    if not dims:
        raise ValueError(f"{value.name} has no shape information")
    dims[0].ClearField("dim_param")
    dims[0].dim_value = int(rows)


def _tensor_rank_from_value(value) -> int | None:
    dims = value.type.tensor_type.shape.dim
    if not dims:
        return None
    return len(dims)


def _collect_constant_arrays(model: onnx.ModelProto) -> dict[str, np.ndarray]:
    consts: dict[str, np.ndarray] = {}
    for init in model.graph.initializer:
        consts[init.name] = numpy_helper.to_array(init)
    for node in model.graph.node:
        if node.op_type != "Constant" or not node.output:
            continue
        for attr in node.attribute:
            if attr.name == "value":
                consts[node.output[0]] = numpy_helper.to_array(helper.get_attribute_value(attr))
                break
    return consts


def _collect_known_ranks(model: onnx.ModelProto) -> dict[str, int]:
    ranks: dict[str, int] = {}
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        rank = _tensor_rank_from_value(value)
        if rank is not None:
            ranks[value.name] = rank
    for init in model.graph.initializer:
        ranks[init.name] = len(init.dims)
    return ranks


def _infer_output_rank(node: onnx.NodeProto, input_ranks: list[int | None], axes_count: int | None = None) -> int | None:
    op = node.op_type
    if op in {"Relu", "Sqrt", "Identity"}:
        return input_ranks[0]
    if op in {"Add", "Sub", "Mul", "Div", "Pow"}:
        known = [rank for rank in input_ranks if rank is not None]
        return max(known) if known else None
    if op == "MatMul":
        if len(input_ranks) >= 2 and input_ranks[0] == 2 and input_ranks[1] == 2:
            return 2
        return input_ranks[0]
    if op == "ReduceMean":
        input_rank = input_ranks[0]
        if input_rank is None:
            return None
        keepdims = 1
        for attr in node.attribute:
            if attr.name == "keepdims":
                keepdims = int(helper.get_attribute_value(attr))
                break
        if keepdims:
            return input_rank
        if axes_count is None:
            return None
        return max(0, input_rank - axes_count)
    return None


def _axes_count_from_attrs(node: onnx.NodeProto) -> int | None:
    for attr in node.attribute:
        if attr.name == "axes":
            return len(helper.get_attribute_value(attr))
    return None


def rewrite_reduce_nodes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    const_arrays = _collect_constant_arrays(model)
    ranks = _collect_known_ranks(model)
    new_nodes = []
    converted = 0

    for node in model.graph.node:
        input_ranks = [ranks.get(name) for name in node.input]
        current = node

        if node.op_type == "ReduceMean" and len(node.input) >= 2 and node.input[1] in const_arrays:
            input_rank = input_ranks[0]
            if input_rank is None:
                raise ValueError(f"Cannot infer rank for ReduceMean input {node.input[0]}")
            axes_raw = np.array(const_arrays[node.input[1]]).astype(np.int64).reshape(-1)
            axes = []
            for axis in axes_raw.tolist():
                axis_i = int(axis)
                if axis_i < 0:
                    axis_i += input_rank
                axes.append(axis_i)
            attrs = {}
            for attr in node.attribute:
                if attr.name == "keepdims":
                    attrs["keepdims"] = int(helper.get_attribute_value(attr))
                elif attr.name == "noop_with_empty_axes":
                    attrs["noop_with_empty_axes"] = int(helper.get_attribute_value(attr))
            attrs["axes"] = axes
            current = helper.make_node(
                "ReduceMean",
                [node.input[0]],
                list(node.output),
                name=node.name,
                **attrs,
            )
            input_ranks = [input_rank]
            converted += 1

        new_nodes.append(current)
        inferred_rank = _infer_output_rank(current, input_ranks, axes_count=_axes_count_from_attrs(current))
        if inferred_rank is not None:
            for output in current.output:
                ranks[output] = inferred_rank

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    return model, converted


def rewrite_gemm_nodes(model: onnx.ModelProto) -> tuple[onnx.ModelProto, int]:
    initializer_map = {init.name: init for init in model.graph.initializer}
    used_names = {node.name for node in model.graph.node}
    used_tensor_names = set()
    for node in model.graph.node:
        used_tensor_names.update(node.input)
        used_tensor_names.update(node.output)
    used_tensor_names.update(initializer_map)

    new_nodes = []
    new_initializers = list(model.graph.initializer)
    converted = 0

    for node in model.graph.node:
        if node.op_type != "Gemm":
            new_nodes.append(node)
            continue

        attrs = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
        alpha = float(attrs.get("alpha", 1.0))
        beta = float(attrs.get("beta", 1.0))
        trans_a = int(attrs.get("transA", 0))
        trans_b = int(attrs.get("transB", 0))

        if trans_a != 0:
            raise ValueError(f"Gemm node {node.name or '<unnamed>'} uses transA=1; unsupported in rewrite")
        if node.input[1] not in initializer_map:
            raise ValueError(f"Gemm node {node.name or '<unnamed>'} weight input {node.input[1]} is not an initializer")
        if len(node.input) >= 3 and node.input[2] and node.input[2] not in initializer_map:
            raise ValueError(f"Gemm node {node.name or '<unnamed>'} bias input {node.input[2]} is not an initializer")

        weight_arr = numpy_helper.to_array(initializer_map[node.input[1]])
        if trans_b:
            weight_arr = np.ascontiguousarray(weight_arr.T)
        else:
            weight_arr = np.ascontiguousarray(weight_arr)
        if alpha != 1.0:
            weight_arr = np.ascontiguousarray(weight_arr * alpha)

        weight_name = f"{node.name or node.output[0]}_qairt_weight"
        while weight_name in used_tensor_names:
            weight_name += "_x"
        used_tensor_names.add(weight_name)
        new_initializers.append(numpy_helper.from_array(weight_arr.astype(np.float32), weight_name))

        matmul_out = node.output[0]
        add_output = node.output[0]
        if len(node.input) >= 3 and node.input[2]:
            matmul_out = f"{node.output[0]}_matmul"
            while matmul_out in used_tensor_names:
                matmul_out += "_x"
            used_tensor_names.add(matmul_out)

        new_nodes.append(
            helper.make_node(
                "MatMul",
                [node.input[0], weight_name],
                [matmul_out],
                name=f"{node.name or node.output[0]}_MatMul",
            )
        )

        if len(node.input) >= 3 and node.input[2]:
            bias_arr = numpy_helper.to_array(initializer_map[node.input[2]])
            if beta != 1.0:
                bias_arr = np.ascontiguousarray(bias_arr * beta)
            bias_name = f"{node.name or node.output[0]}_qairt_bias"
            while bias_name in used_tensor_names:
                bias_name += "_x"
            used_tensor_names.add(bias_name)
            new_initializers.append(numpy_helper.from_array(bias_arr.astype(np.float32), bias_name))
            new_nodes.append(
                helper.make_node(
                    "Add",
                    [matmul_out, bias_name],
                    [add_output],
                    name=f"{node.name or node.output[0]}_Add",
                )
            )
        converted += 1

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    del model.graph.initializer[:]
    model.graph.initializer.extend(new_initializers)
    return model, converted


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite ONNX Gemm nodes into QAIRT-friendlier MatMul/Add nodes and optionally freeze batch size.")
    parser.add_argument("--input", required=True, help="Input ONNX path")
    parser.add_argument("--output", required=True, help="Output ONNX path")
    parser.add_argument("--rows", type=int, help="Optional fixed first dimension for graph input/output")
    args = parser.parse_args()

    src = Path(args.input).resolve()
    dst = Path(args.output).resolve()
    model = onnx.load(str(src), load_external_data=True)

    if args.rows is not None:
        if len(model.graph.input) != 1 or len(model.graph.output) != 1:
            raise ValueError("Expected single-input single-output graph when freezing rows")
        _set_first_dim(model.graph.input[0], args.rows)
        _set_first_dim(model.graph.output[0], args.rows)

    model, converted_gemm = rewrite_gemm_nodes(model)
    model, converted_reduce = rewrite_reduce_nodes(model)
    onnx.save(model, str(dst))
    print(f"rewrote {converted_gemm} Gemm node(s) into MatMul/Add in {dst}")
    print(f"rewrote {converted_reduce} ReduceMean node(s) to inline positive axes")
    if args.rows is not None:
        print(f"fixed batch rows to {args.rows}")


if __name__ == "__main__":
    main()
