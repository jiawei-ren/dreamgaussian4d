import torch

from diffusers import StableVideoDiffusionPipeline, DDIMScheduler
# from diffusers.src.diffusers.pipleines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing
from diffusers.utils import load_image, export_to_video, export_to_gif

from PIL import Image
import numpy as np

import torch.nn as nn
import torch.nn.functional as F


class StableVideoDiffusion:
    def __init__(
        self,
        device,
        fp16=True,
        t_range=[0.02, 0.98],
    ):
        super().__init__()

        self.guidance_type = [
            'sds',
            'pixel reconstruction',
            'latent reconstruction'
        ][1]

        self.device = device
        self.dtype = torch.float16 if fp16 else torch.float32

        # Create model
        pipe = StableVideoDiffusionPipeline.from_pretrained(
            "stabilityai/stable-video-diffusion-img2vid", torch_dtype=torch.float16, variant="fp16"
        )
        pipe.to(device)

        self.pipe = pipe

        self.num_train_timesteps = self.pipe.scheduler.config.num_train_timesteps if self.guidance_type == 'sds' else 25
        self.pipe.scheduler.set_timesteps(self.num_train_timesteps, device=device)  # set sigma for euler discrete scheduling

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.pipe.scheduler.alphas_cumprod.to(self.device)  # for convenience

        self.embeddings = None
        self.image = None
        self.target_cache = None

    @torch.no_grad()
    def get_img_embeds(self, image):
        self.image = Image.fromarray(np.uint8(image*255))

    def encode_image(self, image):
        image = image * 2 -1
        latents = self.pipe._encode_vae_image(image, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=False)
        latents = self.pipe.vae.config.scaling_factor * latents
        return latents
    
    def refine(self,
        pred_rgb,
        steps=25, strength=0.8,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
    ):
        # strength = 0.8
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!

        # latents = []
        # for i in range(batch_size):
        #     latent =  self.encode_image(pred_rgb_512[i:i+1])
        #     latents.append(latent)
        # latents = torch.cat(latents, 0)

        latents = self.encode_image(pred_rgb_512)
        latents = latents.unsqueeze(0)

        if strength == 0:
            init_step = 0
            latents = torch.randn_like(latents)
        else:
            init_step = int(steps * strength)
            latents = self.pipe.scheduler.add_noise(latents, torch.randn_like(latents), self.pipe.scheduler.timesteps[init_step:init_step+1])

        target = self.pipe(
            image=self.image,
            height=512,
            width=512,
            latents=latents,
            denoise_beg=init_step,
            denoise_end=steps,
            output_type='frame', 
            num_frames=batch_size,
            min_guidance_scale=min_guidance_scale,
            max_guidance_scale=max_guidance_scale,
            num_inference_steps=steps,
            decode_chunk_size=1
        ).frames[0]
        target = (target + 1) * 0.5
        target  = target.permute(1,0,2,3)
        return target
            
        # frames = self.pipe(
        #     image=self.image,
        #     height=512,
        #     width=512,
        #     latents=latents,
        #     denoise_beg=init_step,
        #     denoise_end=steps,
        #     num_frames=batch_size,
        #     min_guidance_scale=min_guidance_scale,
        #     max_guidance_scale=max_guidance_scale,
        #     num_inference_steps=steps,
        #     decode_chunk_size=1
        # ).frames[0]
        # export_to_gif(frames, f"tmp.gif")
        # raise
 
    def train_step(
        self,
        pred_rgb,
        step_ratio=None,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
    ):
        
        batch_size = pred_rgb.shape[0]
        pred_rgb = pred_rgb.to(self.dtype)

        # interp to 512x512 to be fed into vae.
        pred_rgb_512 = F.interpolate(pred_rgb, (512, 512), mode="bilinear", align_corners=False)
        # encode image into latents with vae, requires grad!
        # latents = self.pipe._encode_image(pred_rgb_512, self.device, num_videos_per_prompt=1, do_classifier_free_guidance=True)
        latents = self.encode_image(pred_rgb_512)
        latents = latents.unsqueeze(0)

        if step_ratio is not None:
            # dreamtime-like
            # t = self.max_step - (self.max_step - self.min_step) * np.sqrt(step_ratio)
            t = np.round((1 - step_ratio) * self.num_train_timesteps).clip(self.min_step, self.max_step)
            t = torch.full((1,), t, dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, (1,), dtype=torch.long, device=self.device)
        # print(t)

        w = (1 - self.alphas[t]).view(1, 1, 1, 1)


        if self.guidance_type == 'sds':
            # predict the noise residual with unet, NO grad!
            with torch.no_grad():
                t = self.num_train_timesteps - t.item()
                # add noise
                noise = torch.randn_like(latents)
                latents_noisy = self.pipe.scheduler.add_noise(latents, noise, self.pipe.scheduler.timesteps[t:t+1]) # t=0 noise;t=999 clean
                noise_pred = self.pipe(
                    image=self.image,
                    # image_embeddings=self.embeddings, 
                    height=512,
                    width=512,
                    latents=latents_noisy,
                    output_type='noise', 
                    denoise_beg=t,
                    denoise_end=t + 1,
                    min_guidance_scale=min_guidance_scale,
                    max_guidance_scale=max_guidance_scale,
                    num_frames=batch_size,
                    num_inference_steps=self.num_train_timesteps
                ).frames[0]
            
            grad = w * (noise_pred - noise)
            grad = torch.nan_to_num(grad)

            target = (latents - grad).detach()
            loss = 0.5 * F.mse_loss(latents.float(), target, reduction='sum') / latents.shape[1]
            print(loss.item())
            return loss
        
        elif self.guidance_type == 'pixel reconstruction':
            # pixel space reconstruction
            if self.target_cache is None:
                with torch.no_grad():
                    self.target_cache = self.pipe(
                        image=self.image,
                        height=512,
                        width=512,
                        output_type='frame', 
                        num_frames=batch_size,
                        num_inference_steps=self.num_train_timesteps,
                        decode_chunk_size=1
                    ).frames[0]
                    self.target_cache = (self.target_cache + 1) * 0.5
                    self.target_cache  = self.target_cache.permute(1,0,2,3)

            loss = 0.5 * F.mse_loss(pred_rgb_512.float(), self.target_cache.detach().float(), reduction='sum') / latents.shape[1]
            print(loss.item())

            return loss

        elif self.guidance_type == 'latent reconstruction':
            # latent space reconstruction
            if self.target_cache is None:
                with torch.no_grad():
                    self.target_cache = self.pipe(
                        image=self.image,
                        height=512,
                        width=512,
                        output_type='latent', 
                        num_frames=batch_size,
                        num_inference_steps=self.num_train_timesteps,
                    ).frames[0]

            loss = 0.5 * F.mse_loss(latents.float(), self.target_cache.detach().float(), reduction='sum') / latents.shape[1]
            print(loss.item())

            return loss