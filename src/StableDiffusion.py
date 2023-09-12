from src.Setup import Pipe, PreTrainedModel
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os


class StableDiffusion:
    def __init__(self, device="cuda", pipe="prompt2img", seed=None, pretrained_model_registry_path=None, scheduler_function=None):
        assert pipe in Pipe
        self.device = device
        self.seed = seed
        self.generator = None
        if self.seed:
            self.generator = self.get_generator()
        self.pretrained_model_registry_path = PreTrainedModel[pipe.name].value if pretrained_model_registry_path is None else pretrained_model_registry_path
        self.pipe = Pipe[pipe].value.from_pretrained(self.pretrained_model_registry_path, torch_dtype=torch.float16).to(self.device)

        # For optimized resource utilization
        self.pipe.enable_attention_slicing()
        self.pipe.enable_xformers_memory_efficient_attention()

        # Setting up scheduler
        if scheduler_function:
            self.scheduler_function = scheduler_function
            self.pipe.scheduler = self.scheduler_function.from_config(self.pipe.scheduler.config)

    @staticmethod
    def grid_img(imgs, rows=1, cols=3, scale=1):
        assert len(imgs) == rows * cols

        w, h = imgs[0].size
        w, h = int(w*scale), int(h*scale)
        
        grid = Image.new('RGB', size=(cols*w, rows*h))
        grid_w, grid_h = grid.size
        
        for i, img in enumerate(imgs):
            img = img.resize((w,h), Image.ANTIALIAS)
            grid.paste(img, box=(i%cols*w, i//cols*h))
        plt.show(grid)
        return grid
    
    def get_generator(self):
        return torch.Generator(device=self.device).manual_seed(self.seed)
    
    def __call__(self, prompt, **kwargs):
        images = self.pipe(prompt=prompt, **kwargs).images
        if isinstance(prompt, list): return StableDiffusion.grid_img(imgs=images, rows=1, cols=len(prompt))
        return images
    
    def save(self, directory, images):
        assert os.path.isdir(directory)
        if isinstance(images, list):
            for i, img in enumerate(images):
                img.save(os.path.join(directory, os.sep, f"result_{i+1}.png"))
        else:
            from datetime import datetime
            images.save(os.path.join(directory, os.sep, f"result_{datetime.now()}.png"))
