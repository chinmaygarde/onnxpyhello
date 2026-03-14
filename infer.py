import numpy as np
import onnxruntime as ort


def main():
    session = ort.InferenceSession("mnist.onnx")

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    print(f"Input:  {input_name}")
    print(f"Output: {output_name}")


    # Draw a simple "1": a vertical line in the center of a 28x28 image.
    img = np.zeros((1, 1, 28, 28), dtype=np.float32)
    img[0, 0, 4:24, 13:15] = 1.0

    (logits,) = session.run([output_name], {input_name: img})

    predicted = int(np.argmax(logits))
    print(f"Predicted digit: {predicted}")
    print(f"Logits: {logits[0].round(2)}")


if __name__ == "__main__":
    main()
