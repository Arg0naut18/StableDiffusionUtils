import os
import json
from natsort import natsorted
from glob import glob
from src.StableDiffusion import StableDiffusion
from PIL import Image


class Train:
    def __init__(self, images_dir, instance_prompt, class_prompt, num_imgs=10, num_class_images=None, max_num_steps=None, learning_rate=1e-6, lr_warmup_steps=None):
        self.model_sd = "runwayml/stable-diffusion-v1-5"
        self.instance_prompt = instance_prompt
        self.output_dir = f"/content/stable_diffusion_weights/{self.instance_prompt}"
        self.images_dir = images_dir
        self.class_data_dir = os.path.join(images_dir, "/../class_dir")
        assert os.path.isdir(images_dir)
        if not os.path.isfile("train_dreambooth.py"):
            Train.setup()
        
        self.num_imgs = num_imgs
        self.num_class_images = 12 * self.num_imgs if not num_class_images else num_class_images
        self.max_num_steps = 80 * self.num_imgs if not max_num_steps else max_num_steps
        self.learning_rate = learning_rate
        self.lr_warmup_steps = self.max_num_steps//10 if not lr_warmup_steps else lr_warmup_steps

    def concept_list(self):
        self.concepts_list = [
            {
                "instance_prompt": self.instance_prompt,
                "class_prompt": self.class_prompt,
                "instance_data_dir": self.images_dir,
                "class_data_dir": self.class_data_dir
            }
        ]
        Train.create_json(self.concepts_list, self.instance_prompt)

    @staticmethod
    def setup():
        os.system("wget -q https://github.com/ShivamShrirao/diffusers/raw/main/examples/dreambooth/train_dreambooth.py")
        os.system("wget -q https://github.com/ShivamShrirao/diffusers/raw/main/scripts/convert_diffusers_to_original_stable_diffusion.py")

    @staticmethod
    def create_json(iterable, name):
        for c in iterable:
            os.makedirs(c["instance_data_dir"], exist_ok=True)
        
        with open(name) as file:
            json.dump(iterable, file, indent=4)

    def train(self, lr_scheduler="constant"):
        command = f"python3 train_dreambooth.py \
            --pretrained_model_name_or_path={self.model_sd} \
            --pretrained_vae_name_or_path=\"stabilityai/sd-vae-ft-mse\" \
            --output_dir={self.output_dir} \
            --revision=\"fp16\" \
            --with_prior_preservation --prior_loss_weight=1.0 \
            --seed=777 \
            --resolution=512 \
            --train_batch_size=1 \
            --train_text_encoder \
            --mixed_precision=\"fp16\" \
            --use_8bit_adam \
            --gradient_accumulation_steps=1 \
            --learning_rate={self.learning_rate} \
            --lr_scheduler={lr_scheduler} \
            --lr_warmup_steps=80 \
            --num_class_images={self.num_class_images} \
            --sample_batch_size=4 \
            --max_train_steps={self.max_num_steps} \
            --save_interval=10000 \
            --save_sample_prompt={self.instance_prompt} \
            --concepts_list=\"concepts_list.json\""
        
        os.system(command)
        
        self.weights_dir = natsorted(glob(self.output_dir + os.sep + '*'))[-1]
        self.show_trained_images()
        self.save_checkpoint()

    def save_checkpoint(self):
        ckpt_path = self.weights_dir + "/model.ckpt"
        half_arg = "--half"
        os.system(f"python convert_diffusers_to_original_stable_diffusion.py --model_path {self.weights_dir}  --checkpoint_path {ckpt_path} {half_arg}")
    
    def show_trained_images(self):
        weights_folder = self.output_dir
        folders = sorted([f for f in os.listdir(weights_folder) if f != "0"], key = lambda x: int(x))

        imgs_test = []

        for imgs, folder in enumerate(folders):
            folder_path = os.path.join(weights_folder, folder)
            image_folder = os.path.join(folder_path, "samples")
            images = [f for f in os.listdir(image_folder)]

        for i in images:
            img_path = os.path.join(image_folder, i)
            r = Image.open(img_path)
            imgs_test.append(r)

        StableDiffusion.grid_img(imgs_test, rows=1, cols=4, scale=1)

    
