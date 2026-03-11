import argparse
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, checker


def make_input(name="input", shape=(1, 8)):
    return helper.make_tensor_value_info(name, TensorProto.FLOAT, list(shape))


def make_output(name="output", shape=(1, 8), dtype=TensorProto.FLOAT):
    return helper.make_tensor_value_info(name, dtype, list(shape))


def make_float_initializer(name, array):
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.FLOAT,
        dims=array.shape,
        vals=array.flatten().astype(np.float32),
    )


def make_int8_initializer(name, array):
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.INT8,
        dims=array.shape,
        vals=array.flatten().astype(np.int8),
    )


def make_uint8_initializer(name, array):
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.UINT8,
        dims=array.shape,
        vals=array.flatten().astype(np.uint8),
    )


def make_int32_initializer(name, array):
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.INT32,
        dims=array.shape,
        vals=array.flatten().astype(np.int32),
    )


def make_int64_initializer(name, array):
    return helper.make_tensor(
        name=name,
        data_type=TensorProto.INT64,
        dims=array.shape,
        vals=array.flatten().astype(np.int64),
    )


def save_model(path, nodes, inputs, outputs, initializers, opset=18, ir_version=10):
    graph = helper.make_graph(
        nodes=nodes,
        name=path.stem,
        inputs=inputs,
        outputs=outputs,
        initializer=initializers,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", opset)])
    model.ir_version = ir_version
    checker.check_model(model)
    onnx.save(model, path)


def infer_input_shape(value_info, fallback=(1, 8)):
    try:
        dims = value_info.type.tensor_type.shape.dim
        shape = []
        for d in dims:
            if d.HasField("dim_value"):
                shape.append(int(d.dim_value))
            else:
                shape.append(None)
        if any(s is None for s in shape):
            return fallback
        return tuple(shape)
    except Exception:
        return fallback


def build_add():
    inp = make_input()
    out = make_output()
    const = make_float_initializer("const", np.random.randn(1, 8).astype(np.float32))
    node = helper.make_node("Add", ["input", "const"], ["output"])
    return [node], [inp], [out], [const]


def build_sub():
    inp = make_input()
    out = make_output()
    const = make_float_initializer("const", np.random.randn(1, 8).astype(np.float32))
    node = helper.make_node("Sub", ["input", "const"], ["output"])
    return [node], [inp], [out], [const]


def build_mul():
    inp = make_input()
    out = make_output()
    const = make_float_initializer("const", np.random.randn(1, 8).astype(np.float32))
    node = helper.make_node("Mul", ["input", "const"], ["output"])
    return [node], [inp], [out], [const]


def build_div():
    inp = make_input()
    out = make_output()
    const = make_float_initializer("const", (np.random.rand(1, 8).astype(np.float32) + 0.1))
    node = helper.make_node("Div", ["input", "const"], ["output"])
    return [node], [inp], [out], [const]


def build_pow():
    inp = make_input()
    out = make_output()
    const = make_float_initializer("const", np.full((1, 8), 2.0, dtype=np.float32))
    node = helper.make_node("Pow", ["input", "const"], ["output"])
    return [node], [inp], [out], [const]


def build_sqrt():
    inp = make_input()
    out = make_output()
    node = helper.make_node("Sqrt", ["input"], ["output"])
    return [node], [inp], [out], []


def build_reduce_mean():
    inp = make_input()
    out = make_output(shape=(1, 1))
    axes = make_int64_initializer("axes", np.array([1], dtype=np.int64))
    node = helper.make_node("ReduceMean", ["input", "axes"], ["output"], keepdims=1)
    return [node], [inp], [out], [axes]


def build_relu():
    inp = make_input()
    out = make_output()
    node = helper.make_node("Relu", ["input"], ["output"])
    return [node], [inp], [out], []


def build_gemm():
    inp = make_input()
    out = make_output(shape=(1, 4))
    weight = make_float_initializer("weight", np.random.randn(8, 4).astype(np.float32))
    bias = make_float_initializer("bias", np.random.randn(4).astype(np.float32))
    node = helper.make_node("Gemm", ["input", "weight", "bias"], ["output"])
    return [node], [inp], [out], [weight, bias]

def build_gemm_transb1():
    inp = make_input()
    out = make_output(shape=(1, 4))
    weight = make_float_initializer("weight", np.random.randn(4, 8).astype(np.float32))
    bias = make_float_initializer("bias", np.random.randn(4).astype(np.float32))
    node = helper.make_node("Gemm", ["input", "weight", "bias"], ["output"], transB=1)
    return [node], [inp], [out], [weight, bias]


def build_qdq_gemm_int8():
    inp = make_input()
    out = make_output(shape=(1, 4))
    a_scale = make_float_initializer("a_scale", np.array([0.02], dtype=np.float32))
    a_zero = make_int8_initializer("a_zero", np.array([0], dtype=np.int8))
    w_scale = make_float_initializer("w_scale", np.array([0.03], dtype=np.float32))
    w_zero = make_int8_initializer("w_zero", np.array([0], dtype=np.int8))

    weight = make_int8_initializer("weight_q", np.random.randint(-5, 6, size=(8, 4)).astype(np.int8))
    bias = make_float_initializer("bias", np.random.randn(4).astype(np.float32))

    q_in = helper.make_node("QuantizeLinear", ["input", "a_scale", "a_zero"], ["a_q"])
    dq_in = helper.make_node("DequantizeLinear", ["a_q", "a_scale", "a_zero"], ["a_f"])
    dq_w = helper.make_node("DequantizeLinear", ["weight_q", "w_scale", "w_zero"], ["w_f"])
    gemm = helper.make_node("Gemm", ["a_f", "w_f", "bias"], ["output"])
    return [q_in, dq_in, dq_w, gemm], [inp], [out], [a_scale, a_zero, w_scale, w_zero, weight, bias]


def build_qdq_gemm_uint8():
    inp = make_input()
    out = make_output(shape=(1, 4))
    a_scale = make_float_initializer("a_scale", np.array([0.02], dtype=np.float32))
    a_zero = make_uint8_initializer("a_zero", np.array([128], dtype=np.uint8))
    w_scale = make_float_initializer("w_scale", np.array([0.03], dtype=np.float32))
    w_zero = make_uint8_initializer("w_zero", np.array([128], dtype=np.uint8))

    weight = make_uint8_initializer("weight_q", np.random.randint(0, 256, size=(8, 4)).astype(np.uint8))
    bias = make_float_initializer("bias", np.random.randn(4).astype(np.float32))

    q_in = helper.make_node("QuantizeLinear", ["input", "a_scale", "a_zero"], ["a_q"])
    dq_in = helper.make_node("DequantizeLinear", ["a_q", "a_scale", "a_zero"], ["a_f"])
    dq_w = helper.make_node("DequantizeLinear", ["weight_q", "w_scale", "w_zero"], ["w_f"])
    gemm = helper.make_node("Gemm", ["a_f", "w_f", "bias"], ["output"])
    return [q_in, dq_in, dq_w, gemm], [inp], [out], [a_scale, a_zero, w_scale, w_zero, weight, bias]

def build_ln_prim():
    inp = make_input()
    out = make_output()
    axes = make_int64_initializer("axes", np.array([1], dtype=np.int64))
    eps = make_float_initializer("eps", np.array(1e-5, dtype=np.float32))
    gamma = make_float_initializer("gamma", np.ones((8,), dtype=np.float32))
    beta = make_float_initializer("beta", np.zeros((8,), dtype=np.float32))
    two = make_float_initializer("two", np.array(2.0, dtype=np.float32))

    mean = helper.make_node("ReduceMean", ["input", "axes"], ["mean"], keepdims=1)
    sub = helper.make_node("Sub", ["input", "mean"], ["centered"])
    pow_ = helper.make_node("Pow", ["centered", "two"], ["sq"])
    var = helper.make_node("ReduceMean", ["sq", "axes"], ["var"], keepdims=1)
    var_eps = helper.make_node("Add", ["var", "eps"], ["var_eps"])
    denom = helper.make_node("Sqrt", ["var_eps"], ["denom"])
    norm = helper.make_node("Div", ["centered", "denom"], ["norm"])
    scaled = helper.make_node("Mul", ["norm", "gamma"], ["scaled"])
    out_node = helper.make_node("Add", ["scaled", "beta"], ["output"])

    nodes = [mean, sub, pow_, var, var_eps, denom, norm, scaled, out_node]
    inits = [axes, eps, gamma, beta, two]
    return nodes, [inp], [out], inits


def build_block_mlp_ln():
    inp = make_input(shape=(8, 384))
    out = make_output(shape=(8, 128))

    w0 = make_float_initializer("w0", np.random.randn(128, 384).astype(np.float32))
    b0 = make_float_initializer("b0", np.random.randn(128).astype(np.float32))
    w1 = make_float_initializer("w1", np.random.randn(128, 128).astype(np.float32))
    b1 = make_float_initializer("b1", np.random.randn(128).astype(np.float32))
    w2 = make_float_initializer("w2", np.random.randn(128, 128).astype(np.float32))
    b2 = make_float_initializer("b2", np.random.randn(128).astype(np.float32))

    lin0 = helper.make_node("Gemm", ["input", "w0", "b0"], ["x0"], transB=1)
    relu0 = helper.make_node("Relu", ["x0"], ["x1"])
    lin1 = helper.make_node("Gemm", ["x1", "w1", "b1"], ["x2"], transB=1)
    relu1 = helper.make_node("Relu", ["x2"], ["x3"])
    lin2 = helper.make_node("Gemm", ["x3", "w2", "b2"], ["x4"], transB=1)

    axes = make_int64_initializer("axes", np.array([1], dtype=np.int64))
    eps = make_float_initializer("eps", np.array(1e-5, dtype=np.float32))
    gamma = make_float_initializer("gamma", np.ones((128,), dtype=np.float32))
    beta = make_float_initializer("beta", np.zeros((128,), dtype=np.float32))
    two = make_float_initializer("two", np.array(2.0, dtype=np.float32))

    mean = helper.make_node("ReduceMean", ["x4", "axes"], ["mean"], keepdims=1)
    sub = helper.make_node("Sub", ["x4", "mean"], ["centered"])
    pow_ = helper.make_node("Pow", ["centered", "two"], ["sq"])
    var = helper.make_node("ReduceMean", ["sq", "axes"], ["var"], keepdims=1)
    var_eps = helper.make_node("Add", ["var", "eps"], ["var_eps"])
    denom = helper.make_node("Sqrt", ["var_eps"], ["denom"])
    norm = helper.make_node("Div", ["centered", "denom"], ["norm"])
    scaled = helper.make_node("Mul", ["norm", "gamma"], ["scaled"])
    out_node = helper.make_node("Add", ["scaled", "beta"], ["output"])

    nodes = [lin0, relu0, lin1, relu1, lin2, mean, sub, pow_, var, var_eps, denom, norm, scaled, out_node]
    inits = [w0, b0, w1, b1, w2, b2, axes, eps, gamma, beta, two]
    return nodes, [inp], [out], inits


PROBES = {
    "add": (build_add, 18),
    "sub": (build_sub, 18),
    "mul": (build_mul, 18),
    "div": (build_div, 18),
    "pow": (build_pow, 18),
    "sqrt": (build_sqrt, 18),
    "reduce_mean": (build_reduce_mean, 18),
    "relu": (build_relu, 18),
    "gemm": (build_gemm, 18),
    "gemm_transb1": (build_gemm_transb1, 18),
    "layernorm_prim": (build_ln_prim, 18),
    "block_mlp_ln_8x384_128": (build_block_mlp_ln, 18),
    "qdq_gemm_int8": (build_qdq_gemm_int8, 18),
    "qdq_gemm_uint8": (build_qdq_gemm_uint8, 18),
}


def main():
    parser = argparse.ArgumentParser(description="Generate tiny ONNX probe models for NNAPI support checks.")
    parser.add_argument("--assets", required=True, help="Android assets directory")
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    np.random.seed(args.seed)

    assets_dir = Path(args.assets)
    probes_dir = assets_dir / "probes"
    probes_dir.mkdir(parents=True, exist_ok=True)

    probes = []
    for name, (builder, opset) in PROBES.items():
        nodes, inputs, outputs, inits = builder()
        model_path = probes_dir / f"{name}.onnx"
        save_model(model_path, nodes, inputs, outputs, inits, opset=opset, ir_version=10)

        input_shape = list(infer_input_shape(inputs[0]))
        input_data = np.random.randn(*input_shape).astype(np.float32)
        input_bin = probes_dir / f"{name}_input.bin"
        input_shape_json = probes_dir / f"{name}_input_shape.json"
        input_bin.write_bytes(input_data.tobytes())
        input_shape_json.write_text(json.dumps(input_shape))

        probes.append(
            {
                "name": f"probe/{name}",
                "model": f"probes/{name}.onnx",
                "input_bin": f"probes/{name}_input.bin",
                "input_shape": f"probes/{name}_input_shape.json",
            }
        )

    (assets_dir / "probes.json").write_text(json.dumps(probes, indent=2))
    print(f"Wrote {len(probes)} probes to {probes_dir} and probes.json in {assets_dir}")


if __name__ == "__main__":
    main()
