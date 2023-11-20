# GFPGAN

GFP-GAN: Towards Real-World Blind Face Restoration with Generative Facial Prior

## Dependencies

- CUDA-11.8
- cuDNN-8.9.5.29
- TensorRT-8.6.1.6
- cuda-python-12.3.0
- onnx-1.14.0

## Depoly Inswapper

- step 1: generate TensorRT engine

```bash
python onnx_to_trt/build_engine.py --onnx inswapper_128.onnx --engine inswapper_128_fp16.engine -p fp16
```