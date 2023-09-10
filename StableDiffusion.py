import os


class StableDiffusion:
    def __init__(self, device="cuda", mode="prompt2img"):
        self.device = device
        if mode=="img2img":
            

    @staticmethod
    def setup():
        os.system("pip install -qqq git+https://github.com/huggingface/diffusers.git")
        os.system("pip install -q accelerate transformers ftfy bitsandbytes==0.35.0 gradio natsort safetensors xformers")

    