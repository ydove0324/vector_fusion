from diffusers import StableDiffusionPipeline
import torch
from os import path as osp
from utils import check_and_create_dir
# import cv2

model_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16,use_auth_token="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",local_files_only=True)
pipe.to("cuda")

for i in range(0,20):
    print(i,":iteration")
    prompt = "A tall horse next to a red car"
    suffix = "minimal flat 2d vector icon. trending on a artstation. on a white background"
    image = pipe(prompt=f"{prompt}. {suffix}",num_inference_steps=100).images[0]
    filename = osp.join("..","output",f"{prompt.replace(' ','_')}_more_inference_suffix",f"{i}.png")
    check_and_create_dir(filename)
    image.save(filename)