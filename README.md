# da3_onnx
Depth Anything 3 to ONNX convertion

## Summary

This repository contains a convertion script to convert Depth Anything 3 models from their hugging face repository (https://huggingface.co/spaces/depth-anything/depth-anything-3) to ONNX format.

## Pre-requisites

1. Create a virtual environment (python >= 3.12)
2. Clone this repository
```bash
git clone https://github.com/MSch8791/da3_onnx.git
```
3. Install the dependencies with pip
```bash
pip install -r requirements.txt
```

## Modify it for your need

Modify the convert_da3_to_onnx.py script to add the inputs and outputs you need in the converted model. <b>By default the script only export the multi-view images input and the depth maps output.</b>

## How to convert

From a Hugging Face repository :
```bash
python3 src/convert_da3_to_onnx.py --da3model="depth-anything/DA3-SMALL" --output="da3-small.onnx"
```

From a DA3 model on you hard drive :
```bash
python3 src/convert_da3_to_onnx.py --da3model="../da3_small_model/" --output="da3-small.onnx"
```