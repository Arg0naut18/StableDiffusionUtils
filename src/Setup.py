from enum import Enum
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionInstructPix2PixPipeline, StableDiffusionInpaintPipeline, ControlNetModel, StableDiffusionControlNetPipeline, OpenposeDetector
import torch


class Pipe(Enum):
    prompt2img = StableDiffusionPipeline
    img2img = StableDiffusionImg2ImgPipeline
    inpaint = StableDiffusionInpaintPipeline
    pix2pix = StableDiffusionInstructPix2PixPipeline
    controlnetcannymodel = ControlNetModel
    controlnetmodel =  StableDiffusionControlNetPipeline
    openposedetector = OpenposeDetector
    openposemodel = ControlNetModel
    controlpose = StableDiffusionControlNetPipeline


class PreTrainedModel(Enum):
    prompt2img = "CompVis/stable-diffusion-v2-1"
    img2img = "runwayml/stable-diffusion-v1-5"
    inpaint = "runwayml/stable-diffusion-inpainting"
    pix2pix = "timbrooks/instruct-pix2pix"
    controlnetcannymodel = "lllyasviel/sd-controlnet-canny"
    controlnetmodel = "runwayml/stable-diffusion-v1-5"
    openposedetector = "lllyasviel/ControlNet"
    openposemodel = "thibaud/controlnet-sd21-openpose-diffusers"
    controlpose = "stabilityai/stable-diffusion-2-1-base"
    
