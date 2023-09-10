from enum import Enum
from diffusers import 

class Mode(Enum):
    prompt2img = 1
    img2img = 2
    inplant = 3
    pix2pix = 4