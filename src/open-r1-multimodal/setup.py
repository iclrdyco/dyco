# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Adapted from huggingface/transformers: https://github.com/huggingface/transformers/blob/21a2d900eceeded7be9edc445b56877b95eda4ca/setup.py


from setuptools import find_packages, setup

# The attention patches target the Transformers 4.49.0 Qwen2.5-VL API.
setup(
    name="dyco-rl",
    version="0.1.0",
    description="Dynamic cross-modal coordination for visual reasoning",
    license="Apache-2.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.5.1,<2.7",
        "torchvision>=0.20.1,<0.22",
        "transformers==4.49.0",
        "trl==0.17.0",
        "accelerate>=1.2.1,<2",
        "datasets>=3.2.0,<4",
        "deepspeed==0.15.4",
        "peft>=0.14.0,<0.16",
        "huggingface-hub>=0.26.0,<1.0",
        "qwen-vl-utils>=0.0.8,<0.1",
        "einops>=0.8.0",
        "math-verify>=0.5.0,<0.8",
        "packaging>=23.0",
        "safetensors>=0.3.3",
        "sentencepiece>=0.1.99",
        "Pillow>=10.0",
        "numpy>=1.26,<3",
    ],
    extras_require={
        "logging": ["wandb>=0.19.1"],
    },
    zip_safe=False,
)
