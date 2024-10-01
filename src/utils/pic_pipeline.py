import sys
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from torch.optim.optimizer import Optimizer
import math

from random import randrange
from typing import Any, Callable, Dict, List, Optional, Union, Tuple
from diffusers import DDIMScheduler
import time
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
sys.path.insert(0, "src/utils")
from base_pipeline import BasePipeline
from cross_attention import prep_unet
from tqdm.autonotebook import tqdm
import clip
import torchvision.models as models
from pytorch_lightning import seed_everything
# from vgg import Vgg16

from PIL import Image
from utils.scheduler import DDIMInverseScheduler

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

class PicPipeline(BasePipeline):

    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        num_inversion_steps: int = 50,
        guidance_scale_for: float = 7.5,
        guidance_scale_rev: float = 7.5,
        task_name = None,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        cross_attention_kwargs: Optional[Dict[str, Any]] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        img=None, # the input image as a PIL image
        torch_dtype=torch.float32,

        use_wordswap: bool = False,
        use_lowvram: bool = False,
        is_synthetic: bool = False,
        beta: float = 0.0,
        tau: int = 50,
        gamma: float = 0.0,
    ):
        print(f"Use Low VRAM? (Computational speed can be slowed down.): {use_lowvram}")
        
        device = self._execution_device
        seed_everything(42)

        self.unet = prep_unet(self.unet)

        do_classifier_free_guidance = guidance_scale_for > 1.0
        
        self.scheduler = DDIMInverseScheduler.from_config(self.scheduler.config)
        self.scheduler.set_timesteps(num_inversion_steps, device=device)
        timesteps = self.scheduler.timesteps

        negative_prompt = prompt
        
        x0 = np.array(img)/255
        x0 = torch.from_numpy(x0).type(torch_dtype).permute(2, 0, 1).unsqueeze(dim=0).repeat(1, 1, 1, 1).to(device)
        x0 = (x0 - 0.5) * 2. 

        with torch.no_grad():
            x0_enc = self.vae.encode(x0).latent_dist.sample().to(device, torch_dtype)

        latents = x0_enc = 0.18215 * x0_enc
        
        with torch.no_grad():
            x0_dec = self.decode_latents(x0_enc.detach())

        with torch.no_grad():
            prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=None, negative_prompt_embeds=negative_prompt_embeds,)
        
        
        prompt_change, idx_list, is_added = text_swap(task_name, prompt, device)

        print(idx_list)
        print(f'Source Prompt: {prompt}')
        print(f'Target Prompt: {prompt_change}')

        with torch.no_grad():
            prompt_to = self._encode_prompt(prompt_change, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

        extra_step_kwargs = self.prepare_extra_step_kwargs(None, eta)

        latent_save = {}
        eps_save = {}

        num_warmup_steps = len(timesteps) - num_inversion_steps * self.scheduler.order
        
        with self.progress_bar(total=num_inversion_steps) as progress_bar:
            for i, t in enumerate(timesteps.flip(0)[1:-1]): 

                latent_save[t.item()] = latents.detach().clone()

                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
                
                with torch.no_grad():
                    noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=prompt_embeds,cross_attention_kwargs=cross_attention_kwargs,).sample

                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + guidance_scale_for * (noise_pred_text - noise_pred_uncond)
                    
                latents = self.scheduler.step(noise_pred, t, latents, reverse=True, **extra_step_kwargs).prev_sample

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
        
        x_inv = latents.detach().clone()
        num_inference_steps = num_inversion_steps
        self.scheduler = DDIMScheduler.from_config(self.scheduler.config)
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = guidance_scale_rev > 1.0
        
        if is_synthetic:
            x_in = torch.randn((1,4,64,64)).to(dtype=self.unet.dtype, device=self._execution_device)
            prompt = input('[Synthetic Mode] Prompt: ')
            negative_prompt = prompt
            if use_wordswap:
                prompt_change, idx_list = text_swap(task_name, prompt, device)
                # prompt_change = input(f'prompt: {prompt} -> ')
        
                print(f'{prompt} -> {prompt_change}, {idx_list[0]}')
            else:
                prompt_change = input(f'prompt: {prompt} -> ')

        else:
            x_in = x_inv.to(dtype=self.unet.dtype, device=self._execution_device)
        
        del latents, x_inv

        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        num_channels_latents = self.unet.in_channels
        
        latents = self.prepare_latents(batch_size * num_images_per_prompt, num_channels_latents, height, width, prompt_embeds.dtype, device, generator, x_in,)
        latents_init = latents.clone()

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order

        with torch.no_grad():
            prompt_to = self._encode_prompt(prompt_change, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=None, negative_prompt_embeds=negative_prompt_embeds,)
        
        with torch.no_grad():
            prompt_embeds = self._encode_prompt(prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt, prompt_embeds=None, negative_prompt_embeds=negative_prompt_embeds,)

        prompt_embeds_edit = prompt_embeds.clone()
        
        prompt_embeds_edit[1:2] = prompt_to[1:2].clone()
        
        with torch.no_grad():
            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i, t in enumerate(timesteps):
                    
                    
                    alpha = (1-beta) * ((i+1)/tau) + beta  

                    noise_pred = pred_noise(self, latents, t, prompt_embeds[0:1], False, guidance_scale_rev, cross_attention_kwargs, use_lowvram)
                

                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                    if i < tau:
                        eps_save[t.item()] = noise_pred.detach().clone()
                    
        
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        image_rec = self.numpy_to_pil(self.decode_latents(latents.detach()))
        latents = latents_init.clone()

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
        
                alpha = (1-beta) * ((i+1)/tau) + beta
                with torch.no_grad():

                    if i < tau:

                        noise_src = eps_save[t.item()].clone()
                
                        prompt_embeds_star = interpolate_text(alpha, prompt_embeds, prompt_embeds_edit, idx_list).clone()

                        noise_pred_star = pred_noise(self, latents, t, prompt_embeds_star, do_classifier_free_guidance, guidance_scale_rev, cross_attention_kwargs, use_lowvram, batched=True)
                        noise_pred_star_pred, noise_pred_star = noise_pred_star.chunk(2)
                
                        text_gui = (noise_pred_star - noise_pred_star_pred) * guidance_scale_rev
                        noise_delta = gamma * text_gui
                        noise_pred = noise_src + noise_delta
    
                    else:
                        noise_pred = pred_noise(self, latents, t, prompt_embeds_edit, do_classifier_free_guidance, guidance_scale_rev, cross_attention_kwargs, use_lowvram)
                    

                latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample

                with torch.no_grad(): 
                    if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                        progress_bar.update()

        with torch.no_grad():
            image = self.decode_latents(latents.detach())

        x_img = self.numpy_to_pil(image)
        
        return x_img, image_rec


def interpolate_text(scale, src_embeds, tgt_embeds, idx_list, is_added=False):

    assert src_embeds.shape == tgt_embeds.shape

    _src_embeds = src_embeds.clone()
    _tgt_embeds = tgt_embeds.clone()
    temp = tgt_embeds[1].clone()
    
    if not is_added:
        temp = _src_embeds[1].clone() * (1-scale) + _tgt_embeds[1].clone() * scale
    
    else:
        cnt = 0
        for idx, v in enumerate(idx_list):
            if v == "*":
                cnt += 1
            else:
                temp[idx] = scale * _tgt_embeds[1][idx] + (1-scale) * _src_embeds[1][v]
    
    temp = torch.stack([_tgt_embeds[0].clone(), temp], dim = 0)
    return temp.clone()
                        
                        
def pred_noise(self, latents, t, prompt_emb, do_classifier_free_guidance, guidance_scale_rev, cross_attention_kwargs, use_lowvram, batched=False):
    with torch.no_grad():
        if use_lowvram:
            latent_uncond_input = self.scheduler.scale_model_input(latents, t)
            noise_pred_uncond = self.unet(latent_uncond_input,t,encoder_hidden_states=prompt_emb[0:1], cross_attention_kwargs=cross_attention_kwargs,).sample

            latent_cond_input = self.scheduler.scale_model_input(latents, t)
            noise_pred_cond = self.unet(latent_cond_input,t,encoder_hidden_states=prompt_emb[1:2], cross_attention_kwargs=cross_attention_kwargs,).sample

            if batched:
                return torch.cat([noise_pred_uncond, noise_pred_cond], dim=0)
            noise_pred = noise_pred_uncond + guidance_scale_rev * (noise_pred_cond - noise_pred_uncond)

        else:
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            noise_pred = self.unet(latent_model_input,t,encoder_hidden_states=prompt_emb, cross_attention_kwargs=cross_attention_kwargs,).sample
            
            if batched:
                return noise_pred
            latents = latent_model_input.detach().chunk(2)[0]
            
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale_rev * (noise_pred_text - noise_pred_uncond)

        return noise_pred

def text_swap(task, prompt, device):
    word_f, word_t = task.split('2')

    with torch.no_grad():
        from torch.nn import CosineSimilarity
        from transformers import CLIPTokenizer, CLIPModel, CLIPTextModel
        cossim = CosineSimilarity(dim=0, eps=1e-6)

        def dist(v1, v2):
            return cossim(v1, v2)

        tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
        text_encoder = CLIPTextModel.from_pretrained('openai/clip-vit-base-patch32').to(device)

        prompts = list(prompt.split(' '))
        
        text_inputs = tokenizer(
            prompts, 
            padding="max_length", 
            return_tensors="pt",
            ).to(device)

        text_f = tokenizer(
            word_f, 
            padding="max_length", 
            return_tensors="pt",
            ).to(device)

        wordf_embeddings = torch.flatten(text_encoder(text_f.input_ids.to(device))['last_hidden_state'],1,-1)
        text_embeddings = torch.flatten(text_encoder(text_inputs.input_ids.to(device))['last_hidden_state'],1,-1)

        temp = []
        for i in range(len(prompts)): temp.append(dist(text_embeddings[i], wordf_embeddings[0]).item())
        idx_start = temp.index(max(temp))

        prompt_tgt = prompt.replace(prompts[idx_start], word_t, 1)

        idx_list = [i for i in range(77)]
        
        is_added = False if len(word_t.split(' ')) == len(word_f.split(' ')) else True

        cnt = 0

        if is_added:
            idx_list = idx_list[:idx_start+1] + ["*"] * len(word_t.split(' ')) + idx_list[idx_start+1:]
            idx_list = idx_list[:len(prompt_tgt.split(' '))+1]

        else:
            idx_list = idx_list[:idx_start+1] + ["*"] + idx_list[idx_start+2:]
            idx_list = idx_list[:len(prompt_tgt.split(' '))+1]

        return prompt_tgt, idx_list, is_added
        