import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper


def build_model() -> onnx.ModelProto:
    # A simple graph: C = A + B
    add_node = helper.make_node("Add", inputs=["A", "B"], outputs=["C"])

    graph = helper.make_graph(
        nodes=[add_node],
        name="hello-onnx",
        inputs=[
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [3]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [3]),
        ],
        outputs=[
            helper.make_tensor_value_info("C", TensorProto.FLOAT, [3]),
        ],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 20)])
    onnx.checker.check_model(model)
    return model


def main():
    model = build_model()

    session = ort.InferenceSession(model.SerializeToString())

    a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    b = np.array([10.0, 20.0, 30.0], dtype=np.float32)

    (c,) = session.run(["C"], {"A": a, "B": b})

    print(f"A = {a}")
    print(f"B = {b}")
    print(f"C = A + B = {c}")


if __name__ == "__main__":
    main()
