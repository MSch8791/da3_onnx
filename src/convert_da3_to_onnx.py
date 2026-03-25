# Copyright (c) 2026 MSch8791.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import argparse

import torch
import onnx

from depth_anything_3.api import DepthAnything3


class DepthAnything3Wrapper(torch.nn.Module):
    def __init__(self, api_model: DepthAnything3) -> None:
        super().__init__()
        self._model = api_model

    # TODO HERE : you can add the others inputs (intrinsics matrix, extrinsics matrix, etc) and 
    # return the others outputs given by the model
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        model_in = image

        with torch.no_grad():
            dtype = torch.float32 if model_in.device.type == "cpu" else torch.float16
            with torch.autocast(device_type=model_in.device.type, dtype=dtype):
                # we use the internal model object
                output = self._model.model(
                    model_in,
                    extrinsics=None,
                    intrinsics=None,
                    export_feat_layers=[],
                    infer_gs=False,
                    use_ray_pose=False
                )
        
        depth = output["depth"]
        return depth
    
def getArguments():
    parser = argparse.ArgumentParser(description='Replay tool for performance testing')
    parser.add_argument('--da3model', type=str, help='The Depth Anything 3 model to convert. It can be a Hugging Face project identifier or a path to the downloaded model.')
    parser.add_argument('--output', type=str, help='The path where to write the ONNX model file (.onnx).')

    args = parser.parse_args()

    return args

if __name__ == "__main__":
    args = getArguments()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Loading the model...")
    model = DepthAnything3.from_pretrained(args.da3model)
    model = model.to(device)
    model.eval()
    print(f"Model at {args.da3model} loaded on {device}")

    print("Converting...")
    wrapper = DepthAnything3Wrapper(model).to(device)

    B, N_views, C, H, W = 1, 4, 3, 280, 504
    # TODO HERE : add the others dummy inputs necessary (dummy intrinsic/extrinsic/etc tensors)
    dummy_input = torch.zeros(B, N_views, C, H, W).to(device)

    # TODO HERE : add the others input and output names you need
    with torch.no_grad():
        output = wrapper(dummy_input)
        onnx_program = torch.onnx.export(
            wrapper,
            dummy_input,
            do_constant_folding=True,
            input_names=["image"],
            output_names=["depth", "ray"],
            training=torch.onnx.TrainingMode.EVAL,
            dynamo=True, 
            report=True
        )

        # save to converted model
        onnx_program.save(args.output)

    print(f"Convertion done successfully, model save at {args.output}.")

    print("Checking the model...")
    # check the converted model
    onnx_model = onnx.load(args.output)
    onnx.checker.check_model(onnx_model)

    print("Job done.")