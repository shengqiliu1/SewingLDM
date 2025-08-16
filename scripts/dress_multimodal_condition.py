import os
import sys
from pathlib import Path
current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))
import warnings
warnings.filterwarnings("ignore")  # ignore warning
import re
import argparse
from datetime import datetime
from tqdm import tqdm
import torch
import torchvision
import json
import numpy as np
from PIL import Image, ImageOps
from torchvision import transforms as T
from transformers import T5EncoderModel, T5Tokenizer

from diffusion.data.builder import build_dataset
from diffusion.utils.misc import read_config
from diffusion import IDDPM, DPMS, SASolverSampler
from diffusion.model.nets import PixArt_XL_2, MultiModalNet, SewingLDM
from diffusion.model.builder import build_model
import shutil


def get_chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def find_model(model_name):
    assert os.path.isfile(model_name), f'Could not find checkpoint at {model_name}'
    return torch.load(model_name, map_location=lambda storage, loc: storage)
    

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default='configs/random_body/img512_garment_code.py', help="config")
    parser.add_argument('--image_size', default=1024, type=int)
    parser.add_argument(
        "--t5_load_from", default='PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder, "
                       "tokenizer and vae from https://huggingface.co/PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers"
    )
    parser.add_argument('--txt_file', default='assets/samples.txt', type=str)
    parser.add_argument('--model_path', default='output/your_first_pixart-exp/checkpoints/epoch_100_step_24601.pth', type=str)
    parser.add_argument('--bs', default=1, type=int)
    parser.add_argument('--cfg_scale', default=4.5, type=float)
    parser.add_argument('--sampling_algo', default='iddpm', type=str, choices=['iddpm', 'dpm-solver', 'sa-solver'])
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--dataset', default='custom', type=str)
    parser.add_argument('--step', default=-1, type=int)
    parser.add_argument('--save_name', default='test_sample', type=str)
    parser.add_argument('--body_name', default='00463', type=str)
    parser.add_argument('--body_path', default='assets/body_examples/00463_meta.json', type=str)
    parser.add_argument('--sketch_front', default='assets/examples/garment_1/front_sketch.png', type=str)
    parser.add_argument('--sketch_back', default='assets/examples/garment_1/back_sketch.png', type=str)
    parser.add_argument('--txt_path', default='assets/examples/garment_1/caption.txt', type=str)
    parser.add_argument('--idx', type=int, default=0)
    parser.add_argument('--use_text', action='store_true')
    parser.add_argument('--use_sketch', action='store_true')

    return parser.parse_args()


def set_env(seed=0):
    torch.manual_seed(seed)
    torch.set_grad_enabled(False)

@torch.inference_mode()
def visualize(sample_steps, cfg_scale, sketch_front, sketch_back, body_path, text_path):
    if args.use_sketch:
        sketch_threshold = 150
        sketch_front = Image.open(sketch_front)
        sketch_front = ImageOps.invert(sketch_front)
        sketch_front = sketch_front.point(lambda p: 255 if p > sketch_threshold else 0)
        sketch_front = T.functional.to_tensor(sketch_front)
        sketch_back = Image.open(sketch_back)
        sketch_back = ImageOps.invert(sketch_back)
        sketch_back = sketch_back.point(lambda p: 255 if p > sketch_threshold else 0)
        sketch_back = T.functional.to_tensor(sketch_back)

        sketch = torch.cat([sketch_front, sketch_back], dim=0)
        sketch = torchvision.transforms.functional.resize(
            sketch, size=(512, 512),
            interpolation=torchvision.transforms.InterpolationMode.BILINEAR,
            antialias=True)
        sketch = sketch.repeat(1, 1, 1, 1) # or null

        null_sketch = torch.zeros_like(sketch)
        sketch = torch.cat([sketch, null_sketch], dim=0).to(weight_dtype).to(device)
    else:
        sketch = torch.zeros(1, 2, 512, 512).to(weight_dtype).to(device)

    with open(body_path, "r") as f_json:
        body_params = json.load(f_json)['pca_weights']
    body_params = np.array(body_params).astype(np.float32)
    body_params = 0.5 * torch.tensor(body_params).unsqueeze(0).unsqueeze(0).repeat(2, 1, 1).to(weight_dtype).to(device)
    c = {'sketch': sketch, 'body_params': body_params}

    null_y = null_caption_embs.repeat(1, 1, 1)[:, None]
    if args.use_text:
        with open(text_path, 'r') as f:
            prompt = [item.strip() for item in f.readlines()][0]
        caption_token = text_tokenizer(prompt, max_length=max_sequence_length, padding="max_length", truncation=True,
                            return_tensors="pt").to(device)
        caption_embs = text_encoder(caption_token.input_ids, attention_mask=caption_token.attention_mask)[0].unsqueeze(0)
        emb_masks = caption_token.attention_mask.unsqueeze(0)
    else:
        caption_embs = null_y
        emb_masks = None
    print(f'finish embedding')

    with torch.no_grad():
        if args.sampling_algo == 'iddpm':
            # Create sampling noise:
            n = 1
            z = torch.randn(n, config.model_token_size, config.in_channels, device=device).repeat(2, 1, 1)
            model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                cfg_scale=cfg_scale, mask=emb_masks, c=c)
            diffusion = IDDPM(str(sample_steps))
            # Sample images:
            samples = diffusion.p_sample_loop(
                model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
                device=device
            )
            samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        elif args.sampling_algo == 'dpm-solver':
            # Create sampling noise:
            n = 1
            z = torch.randn(n, config.model_token_size, config.in_channels, device=device)
            model_kwargs = dict(data_info={}, mask=emb_masks, c=c)
            dpm_solver = DPMS(model.forward_with_dpmsolver,
                                condition=caption_embs,
                                uncondition=null_y,
                                cfg_scale=cfg_scale,
                                model_kwargs=model_kwargs)
            samples = dpm_solver.sample(
                z,
                steps=sample_steps,
                order=2,
                skip_type="time_uniform",
                method="multistep",
            )
        elif args.sampling_algo == 'sa-solver':
            # Create sampling noise:
            n = 1
            model_kwargs = dict(data_info={}, mask=emb_masks, c=c)
            sa_solver = SASolverSampler(model.forward_with_dpmsolver, device=device)
            samples = sa_solver.sample(
                S=25,
                batch_size=n,
                shape=(config.model_token_size, config.in_channels),
                eta=1,
                conditioning=caption_embs,
                unconditional_conditioning=null_y,
                unconditional_guidance_scale=cfg_scale,
                model_kwargs=model_kwargs,
            )[0]
        if config.use_tokenizer:
            samples = tokenizer.quantize(samples.detach() / config.scaling_factor)
            samples = tokenizer.decode_vector(samples)

        samples = samples.to(weight_dtype)
        max_panel_num = config.start_config['max_pattern_len']
        max_edge_num = config.start_config['max_panel_len']
        recovered_cloth = samples.reshape(1, max_panel_num, max_edge_num, -1)
        discrete = recovered_cloth[..., -10:]
        continuous = recovered_cloth[..., :-10]
        torch.cuda.empty_cache()
        # Save images:
        os.umask(0o000)  # file permission: 666; dir permission: 777
        
        predict_cloths_info = {
            'rotations': continuous[0, ..., -7:-3].mean(-2), # max_pattern_len, 4
            'translations': continuous[0, ..., -3:].mean(-2), # max_pattern_len, 3
            'outlines': continuous[0, ..., :9], # max_pattern_len, max_edge_num, 9-element_size
            'stitch_tags': continuous[0, ..., 9:12], # max_pattern_len, max_edge_num, 3
            'edge_mask' : discrete[0, ..., -1], # max_pattern_len, max_edge_num
            'reverse_stitch': discrete[0, ..., -2].unsqueeze(-1), # max_pattern_len, max_edge_num
            'free_edges_mask': -discrete[0, ..., -3], # max_pattern_len, max_edge_num
            'edge_type': discrete[0, ..., :-3], # max_pattern_len, max_edge_num, 4
        }
        try:
            _, predict_mask_path = dataset.save_prediction_single(
                predict_cloths_info,
                save_to=save_root,
                return_stitches=True)
        except Exception as e:
            print(f"Error {e} in saving prediction, skipping saving")


if __name__ == '__main__':
    args = get_args()
    config = read_config(args.config)
    image_size = config.image_size
    max_length = config.model_max_length
    # Setup PyTorch:
    seed = args.seed
    set_env(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    assert args.sampling_algo in ['iddpm', 'dpm-solver', 'sa-solver']

    # only support fixed latent size currently
    max_sequence_length = 120
    pe_interpolation = {256: 0.5, 512: 1, 1024: 2, 2048: 4}     # trick for positional embedding interpolation
    sample_steps_dict = {'iddpm': 100, 'dpm-solver': 20, 'sa-solver': 25}
    sample_steps = args.step if args.step != -1 else sample_steps_dict[args.sampling_algo]
    weight_dtype = torch.float32
    print(f"Inference with {weight_dtype}")

    tokenizer = None
    if config.use_tokenizer:
        tokenizer = build_model(config.tokenizer,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_token_num=config.tokenizer_token_size,
                        tokenizer=config.tokenizer_config,
                        input_channels=config.continous_channels+config.discrete_channels).eval()
        tokenizer = tokenizer.to(device)
        if config.load_tokenizer_from is not None:
            state_dict = torch.load(config.load_tokenizer_from, map_location=lambda storage, loc: storage)
            missing, unexpected = tokenizer.load_state_dict(state_dict['state_dict'], strict=False)
            print(f'Missing keys: {missing}')
            print(f'Unexpected keys: {unexpected}')

    config.data['load_probability'] = 1
    # build dataset for saving prediction
    config.data['data_json'] = './data_info/test_info.json'
    dataset = build_dataset(
        config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
        real_prompt_ratio=config.real_prompt_ratio, max_length=max_length, config=config,
    )
    # model setting
    dit = PixArt_XL_2(
        token_size=config.model_token_size,
        in_channels=config.in_channels,
        pe_interpolation=pe_interpolation[args.image_size],
        model_max_length=max_sequence_length,
    ).to(device)
    multimodal_net = MultiModalNet(token_size=config.model_token_size).to(device)

    model = SewingLDM(dit, multimodal_net, control_scale=config.control_scale, idx=args.idx).to(device)

    print("Generating sample from ckpt: %s" % args.model_path)
    state_dict = find_model(args.model_path)['state_dict']
    if 'controlnet.pos_embed' in state_dict:
        del state_dict['controlnet.pos_embed']
    if 'base_model.pos_embed' in state_dict:
        del state_dict['base_model.pos_embed']
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print('Missing keys: ', missing)
    print('Unexpected keys', unexpected)
    model.eval()
    model.to(weight_dtype)

    text_tokenizer = T5Tokenizer.from_pretrained(args.t5_load_from, subfolder="tokenizer")
    text_encoder = T5EncoderModel.from_pretrained(args.t5_load_from, subfolder="text_encoder").to(device)

    null_caption_token = text_tokenizer("", max_length=max_sequence_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    null_caption_embs = text_encoder(null_caption_token.input_ids, attention_mask=null_caption_token.attention_mask)[0]

    # img save setting
    img_save_dir = 'output/test'
    os.umask(0o000)  # file permission: 666; dir permission: 777
    os.makedirs(img_save_dir, exist_ok=True)

    save_root = os.path.join(img_save_dir, f"{datetime.now().date()}_{args.dataset}_scale{args.cfg_scale}_step{sample_steps}_samp{args.sampling_algo}_seed{seed}")

    save_root += '_use_text' if args.use_text else '_no_text'
    save_root += f'_use_sketch' if args.use_sketch else f'_no_sketch'
    os.makedirs(save_root, exist_ok=True)
    visualize(sample_steps, args.cfg_scale, args.sketch_front, args.sketch_back, args.body_path, args.txt_path)
