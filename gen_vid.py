import torch

from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif

from PIL import Image
import numpy as np

import cv2
import rembg

import argparse

def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def resize_image(image, output_size=(1024, 576)):
    image = image.resize((output_size[1],output_size[1]))
    pad_size = (output_size[0]-output_size[1]) //2
    image = add_margin(image, 0, pad_size, 0, pad_size, tuple(np.array(image)[0,0]))
    return image


def load_image(file, W, H, bg='white'):
    # load image
    print(f'[INFO] load image from {file}...')
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    bg_remover = rembg.new_session()
    img = rembg.remove(img, session=bg_remover)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    input_mask = img[..., 3:]
    # white bg
    if bg == 'white':
        input_img = img[..., :3] * input_mask + (1 - input_mask)
    elif bg == 'black':
        input_img = img[..., :3]
    else:
        raise NotImplementedError
    # bgr to rgb
    input_img = input_img[..., ::-1].copy()
    input_img = Image.fromarray(np.uint8(input_img*255))
    return input_img

def load_image_w_bg(file, W, H):
    # load image
    print(f'[INFO] load image from {file}...')
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
    img = img.astype(np.float32) / 255.0
    input_img = img[..., :3]
    # bgr to rgb
    input_img = input_img[..., ::-1].copy()
    input_img = Image.fromarray(np.uint8(input_img*255))
    return input_img

def gen_vid(name, seed, bg, is_pad):
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
    )
    # pipe.enable_model_cpu_offload()
    pipe.to("cuda")

    if is_pad:
        height, width = 576, 1024
    else:
        height, width = 512, 512

    if seed is None:
        for bg in ['white', 'black', 'orig']:
            if bg == 'orig':
                if 'rgba' in name:
                    continue
                image = load_image_w_bg(f'data/{name}.png', width, height)
            else:
                image = load_image(f'data/{name}.png', width, height, bg)
            if is_pad:
                image = resize_image(image, output_size=(width, height))
            for seed in range(20):
                generator = torch.manual_seed(seed)
                frames = pipe(image, height, width, generator=generator).frames[0]
                export_to_gif(frames, f"data/videos/{name}_{bg}_{seed:03}.gif")
    else:
        if bg == 'orig':
            if 'rgba' in name:
                raise ValueError
            image = load_image_w_bg(f'data/{name}.png', width, height)
        else:
            image = load_image(f'data/{name}.png', width, height, bg)
        if is_pad:
            image = resize_image(image, output_size=(width, height))
        generator = torch.manual_seed(seed)
        frames = pipe(image, height, width, generator=generator).frames[0]

        export_to_video(frames, f"data/{name}_generated.mp4", fps=7)
        export_to_gif(frames, f"data/{name}_generated.gif")
        for idx, img in enumerate(frames):
            if is_pad:
                img = img.crop(((width-height) //2, 0, width - (width-height) //2, height))
            img.save(f"data/{name}_{idx:03}.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, required=True)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--bg", type=str, default='white')
    parser.add_argument("--is_pad", type=bool, default=False)
    args, extras = parser.parse_known_args()
    gen_vid(args.name, args.seed, args.bg, args.is_pad)
