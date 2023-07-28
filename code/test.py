from diffusers import StableDiffusionPipeline
import torch

model_path = "runwayml/stable-diffusion-v1-5"
pipe = StableDiffusionPipeline.from_pretrained(model_path, torch_dtype=torch.float16,use_auth_token="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",local_files_only=False)
pipe.to("cuda")


image_set = pipe(prompt="Eiffel Tower. minimal flat 2d vector icon. lineal color. on a white background. trending on artstation").images
print(image_set)
image = image_set[0]
image.save("../output/Tower_2.png")