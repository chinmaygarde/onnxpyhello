main:
  uv run main.py

download:
  curl -fL -o mnist.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mnist/model/mnist-12.onnx

infer: download
  uv run infer.py
