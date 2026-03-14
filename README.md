# onnxpyhello

Toying with ONNX models using the Python runtime.

## MNIST Model (`mnist-12.onnx`)

A convolutional neural network trained on the [MNIST dataset](http://yann.lecun.com/exdb/mnist/) to classify handwritten digits 0–9. This is opset 12 from the [ONNX Model Zoo](https://github.com/onnx/models).

**What it does:** Given a grayscale image of a handwritten digit, it returns a score for each of the 10 digit classes. The class with the highest score is the predicted digit.

**Input**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `Input3` | `[1, 1, 28, 28]` | float32 | Single-channel 28×28 image, pixel values normalized to [0.0, 1.0] |

**Output**

| Name | Shape | Type | Description |
|------|-------|------|-------------|
| `Plus214_Output_0` | `[1, 10]` | float32 | Raw logits for digits 0–9. Apply argmax to get the predicted digit. |