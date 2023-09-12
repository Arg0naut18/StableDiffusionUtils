from enum import Enum
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, StableDiffusionInpaintPipeline, EulerAncestralDiscreteScheduler
import torch


class Pipe(Enum):
    prompt2img = StableDiffusionPipeline
    img2img = StableDiffusionImg2ImgPipeline
    inpaint = StableDiffusionInpaintPipeline
    pix2pix = StableDiffusionInstructPix2PixPipeline


class PreTrainedModel(Enum):
    prompt2img = "CompVis/stable-diffusion-v2-1"
    img2img = "runwayml/stable-diffusion-v1-5"
    inpaint = "runwayml/stable-diffusion-inpainting"
    pix2pix = "timbrooks/instruct-pix2pix"
