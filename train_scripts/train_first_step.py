import argparse
import datetime
import os
import sys
import time
import types
import warnings
warnings.filterwarnings("ignore")  # ignore warning
from pathlib import Path

current_file_path = Path(__file__).resolve()
sys.path.insert(0, str(current_file_path.parent.parent))

import numpy as np
import einops
import torch
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.utils import DistributedType
from transformers import T5EncoderModel, T5Tokenizer
from mmcv.runner import LogBuffer
from PIL import Image

from diffusion import IDDPM, DPMS
from diffusion.data.builder import build_dataset, build_datawrapper, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.data_sampler import AspectRatioBatchSampler
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr



def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = 'PixArtBlock'


@torch.inference_mode()
def log_validation(model, step, device, save_folder=None, cfg_scale=2, ae=None):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()
    null_y = torch.load(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
    null_y = null_y['uncond_prompt_embeds'].to(device)

    # Create sampling noise:
    logger.info("Running validation... ")
    image_logs = []
    latents = []

    for prompt in validation_prompts:
        z = torch.randn(1, config.model_token_size, config.in_channels,  device=device).repeat(2, 1, 1)
        embed = torch.load(f'output/tmp/{prompt}_{max_length}token.pth', map_location='cpu')
        caption_embs, emb_masks = embed['caption_embeds'].to(device), embed['emb_mask'].to(device)
        model_kwargs = dict(y=torch.cat([caption_embs, null_y]),
                                    cfg_scale=cfg_scale, mask=emb_masks)

        diffusion = IDDPM(str(100))

        samples = diffusion.p_sample_loop(
            model.forward_with_cfg, z.shape, z, clip_denoised=False, model_kwargs=model_kwargs, progress=True,
            device=device
        )
        samples, _ = samples.chunk(2, dim=0)  # Remove null class samples
        if config.use_tokenizer:
            samples = ae.quantize(samples.detach() / config.scaling_factor)
            samples = ae.decode_vector(samples)
        latents.append(samples)

    torch.cuda.empty_cache()
    datawrapper.dataset.set_training(False)
    max_panel_num = config.start_config['max_pattern_len']
    max_edge_num = config.start_config['max_panel_len']
    for prompt, latent in zip(validation_prompts, latents):
        latent = latent.to(torch.float16).squeeze(0)

        recovered_cloth = latent.reshape(max_panel_num, max_edge_num, -1)
        discrete = recovered_cloth[..., -10:]
        continuous = recovered_cloth[..., :-10]
        predict_cloths_info = {
            'rotations': continuous[..., -7:-3].mean(-2), # max_pattern_len, 4
            'translations': continuous[..., -3:].mean(-2), # max_pattern_len, 3
            'outlines': continuous[..., :9], # max_pattern_len, max_edge_num, 9-element_size
            'stitch_tags': continuous[..., 9:12], # max_pattern_len, max_edge_num, 3
            'edge_mask' : discrete[..., -1], # max_pattern_len, max_edge_num
            'reverse_stitch': discrete[..., -2].unsqueeze(-1), # max_pattern_len, max_edge_num
            'free_edges_mask': -discrete[..., -3], # max_pattern_len, max_edge_num
            'edge_type': discrete[..., :-3], # max_pattern_len, max_edge_num, 4
        }
        cloth_save_folder = os.path.join(save_folder+'_'+str(step), prompt)
        try:
            _, predict_mask_path = dataset.save_prediction_single(
                predict_cloths_info,
                dataname=prompt,
                save_to=cloth_save_folder,
                return_stitches=True)
        except Exception as e:
            logger.warning(f"Error in saving {prompt}: {e}")
            continue

    # flush()
    return image_logs


def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step + 1
    max_panel_num = config.data.start_config['max_pattern_len']
    max_edge_num = config.data.start_config['max_panel_len']

    load_tokenizer_feat = getattr(datawrapper.dataset, 'load_tokenizer_feat', False)
    load_text_feat = getattr(datawrapper.dataset, 'load_text_feat', False)
    condition = getattr(datawrapper.dataset, 'condition', 'text')
    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            if step < skip_step:
                global_step += 1
                continue    # skip data in the resumed ckpt
            cloths_info = batch['ground_truth']
            y_condition = batch['condition']
            bs = len(batch['data_folder'])

            mask = None
            assert condition == 'text'
            if load_text_feat:
                y = y_condition['caption_feature']
                mask = y_condition['attention_mask'].to(torch.int16)
            else:
                with torch.no_grad():
                    txt_tokens = text_tokenizer(
                        y_condition['caption'], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                    ).to(accelerator.device)
                    y = text_encoder(
                        txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                    mask = txt_tokens.attention_mask[:, None, None]

            if config.use_tokenizer:
                if not load_tokenizer_feat:
                    pattern = cloths_info['outlines']
                    stitch_tags = cloths_info['stitch_tags']
                    rots = cloths_info['rotations'].unsqueeze(-2).repeat(1, 1, max_edge_num, 1)
                    tranls = cloths_info['translations'].unsqueeze(-2).repeat(1, 1, max_edge_num, 1)
                    gt_seq = torch.concat([pattern, stitch_tags, rots, tranls], dim=-1)

                    edge_type = cloths_info['edge_type'] # bs, max_panel_num, max_edge_num, 4
                    edge_mask = (~cloths_info['empty_edges_mask']).int().unsqueeze(-1) # bs, max_panel_num, max_edge_num, 1
                    stitch_per_edge = (~cloths_info['free_edges_mask']).int().unsqueeze(-1) # bs, max_panel_num, max_edge_num, 1
                    reverse_stitch = cloths_info['reverse_stitch'].int()
                    discrete_quantization = torch.concat([edge_type, stitch_per_edge, reverse_stitch, edge_mask], dim=-1) # bs, max_panel_num, max_edge_num, 6
                    discrete_quantization = discrete_quantization * 2 - 1
                    input_data = torch.concat([gt_seq, discrete_quantization], dim=-1)
                    input_data = einops.rearrange(input_data, 'b p e c -> b (p e) c')
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(enabled=(config.mixed_precision == 'fp16' or config.mixed_precision == 'bf16')):
                            x_input = ae.encode_vector(input_data) * config.scaling_factor
                else:
                    x_input = cloths_info['latent'] * config.scaling_factor
            else:
                x_input =  input_data

            # Sample a random timestep for each image
            timesteps = torch.randint(0, config.train_sampling_steps, (bs,), device=accelerator.device).long()
            grad_norm = None
            data_time_all += time.time() - data_time_start
            with accelerator.accumulate(model):
                # Predict the noise residual
                optimizer.zero_grad()
                loss_term, _ = train_diffusion.training_losses(model, x_input, timesteps, model_kwargs=dict(y=y, mask=mask))
                loss = loss_term['loss'].mean()

                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            logs.update(diffusion_loss=accelerator.gather(loss_term['loss']).mean().item())

            if grad_norm is not None:
                logs.update(grad_norm=accelerator.gather(grad_norm).mean().item())
            log_buffer.update(logs)
            if (step + 1) % config.log_interval == 0 or (step + 1) == 1:
                t = (time.time() - last_tic) / config.log_interval
                t_d = data_time_all / config.log_interval
                avg_time = (time.time() - time_start) / (global_step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                eta_epoch = str(datetime.timedelta(seconds=int(avg_time * (len(train_dataloader) - step - 1))))
                log_buffer.average()

                info = f"Step/Epoch [{global_step}/{epoch}][{step + 1}/{len(train_dataloader)}]:total_eta: {eta}, " \
                    f"epoch_eta:{eta_epoch}, time_all:{t:.3f}, time_data:{t_d:.3f}, lr:{lr:.3e}, "
                info += f's:({model.module.token_size}, {model.module.in_channels}), ' if hasattr(model, 'module') else f's:({model.token_size}, {model.in_channels}), '

                info += ', '.join([f"{k}:{v:.4f}" for k, v in log_buffer.output.items()])
                logger.info(info)
                last_tic = time.time()
                log_buffer.clear()
                data_time_all = 0
            logs.update(lr=lr)
            accelerator.log(logs, step=global_step)

            global_step += 1
            data_time_start = time.time()

            if global_step % config.save_model_steps == 0:
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    os.umask(0o000)
                    save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                    epoch=epoch,
                                    step=global_step,
                                    model=accelerator.unwrap_model(model),
                                    optimizer=optimizer,
                                    lr_scheduler=lr_scheduler
                                    )
            if config.visualize and (global_step % config.eval_sampling_steps == 0):
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    log_validation(model, global_step, device=accelerator.device, save_folder=os.path.join(config.work_dir, 'visualize'), ae=ae)

        if epoch % config.save_model_epochs == 0 or epoch == config.num_epochs:
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                os.umask(0o000)
                save_checkpoint(os.path.join(config.work_dir, 'checkpoints'),
                                epoch=epoch,
                                step=global_step,
                                model=accelerator.unwrap_model(model),
                                optimizer=optimizer,
                                lr_scheduler=lr_scheduler
                                )
        accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="Process some integers.")
    parser.add_argument("config", type=str, help="config")
    parser.add_argument("--cloud", action='store_true', default=False, help="cloud or local machine")
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the dir to resume the training')
    parser.add_argument('--load-from', default=None, help='the dir to load a ckpt for training')
    parser.add_argument('--local-rank', type=int, default=-1)
    parser.add_argument('--local_rank', type=int, default=-1)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument(
        "--t5_load_from", default='PixArt-alpha/pixart_sigma_sdxlvae_T5_diffusers',
        type=str, help="Download for loading text_encoder"
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="text2garments",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers"
        ),
    )
    parser.add_argument("--loss_report_name", type=str, default="loss")
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    config = read_config(args.config)
    if args.work_dir is not None:
        config.work_dir = args.work_dir
    if args.resume_from is not None:
        config.load_from = None
        config.resume_from = dict(
            checkpoint=args.resume_from,
            load_ema=False,
            resume_optimizer=True,
            resume_lr_scheduler=True)
    if args.debug:
        config.log_interval = 1
        config.train_batch_size = 2
        config.eval_sampling_steps = 2
        config.num_workers = 0
    else:
        torch.multiprocessing.set_start_method('spawn')

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False),)
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True
    if config.multi_scale:
        even_batches=False,

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler]
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name))

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size
    pred_sigma = getattr(config, 'pred_sigma', True)
    learn_sigma = getattr(config, 'learn_sigma', True) and pred_sigma
    max_length = config.model_max_length
    kv_compress_config = config.kv_compress_config if config.kv_compress else None
    ae = None
    if config.use_tokenizer:
        ae = build_model(config.tokenizer,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_token_num=config.tokenizer_token_size,
                        tokenizer=config.tokenizer_config,
                        input_channels=config.continous_channels+config.discrete_channels).eval()
        ae = ae.to(accelerator.device)
        if config.load_tokenizer_from is not None:
            state_dict = torch.load(config.load_tokenizer_from, map_location=lambda storage, loc: storage)
            missing, unexpected = ae.load_state_dict(state_dict['state_dict'], strict=False)
            logger.warning(f'Missing keys: {missing}')
            logger.warning(f'Unexpected keys: {unexpected}')
    if not config.data.load_text_feat:
        text_tokenizer = T5Tokenizer.from_pretrained(args.t5_load_from, subfolder="tokenizer")
        text_encoder = T5EncoderModel.from_pretrained(
            args.t5_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)

    logger.info(f"tokenizer scale factor: {config.scaling_factor}")

    if config.visualize:
        # preparing embeddings for visualization. We put it here for saving GPU memory
        validation_prompts = config.validation_prompts
        skip = True
        Path('output/tmp').mkdir(parents=True, exist_ok=True)
        for prompt in validation_prompts:
            if not (os.path.exists(f'output/tmp/{prompt}_{max_length}token.pth')
                    and os.path.exists(f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')):
                skip = False
                logger.info("Preparing Visualization prompt embeddings...")
                break
        if accelerator.is_main_process and not skip:
            text_tokenizer = T5Tokenizer.from_pretrained(args.t5_load_from, subfolder="tokenizer")
            text_encoder = T5EncoderModel.from_pretrained(
                args.t5_load_from, subfolder="text_encoder", torch_dtype=torch.float16).to(accelerator.device)
            for prompt in validation_prompts:
                txt_tokens = text_tokenizer(
                    prompt, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
                ).to(accelerator.device)
                caption_emb = text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                torch.save(
                    {'caption_embeds': caption_emb, 'emb_mask': txt_tokens.attention_mask},
                    f'output/tmp/{prompt}_{max_length}token.pth')
                del txt_tokens
                del caption_emb
            null_tokens = text_tokenizer(
                "", max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(accelerator.device)
            null_token_emb = text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
            os.makedirs('output/pretrained_models', exist_ok=True)
            torch.save(
                {'uncond_prompt_embeds': null_token_emb, 'uncond_prompt_embeds_mask': null_tokens.attention_mask},
                f'output/pretrained_models/null_embed_diffusers_{max_length}token.pth')
            del null_tokens
            del null_token_emb
            if config.data.load_text_feat:
                del text_encoder
                del text_tokenizer
            flush()
    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(
        config.data, resolution=image_size, max_length=max_length, config=config
    )
    known_split = {'filename': config.data.known_split}
    datawrapper = build_datawrapper(config.datawrapper, dataset, num_workers=config.num_workers, known_split=known_split, batch_size=config.train_batch_size, validation_size=config.validation_size, shuffle=True)
    train_dataloader = datawrapper.get_loader('train')
    valid_dataloader = datawrapper.get_loader('validation')

    model_kwargs = {"pe_interpolation": config.pe_interpolation, "config": config,
                    "model_max_length": max_length, "qk_norm": config.qk_norm, 'caption_channels': 4096,
                    "kv_compress_config": kv_compress_config, "micro_condition": config.micro_condition}

    # build models
    config.data['start_config'].update(standardize=dataset.config['standardize'])
    train_diffusion = IDDPM(str(config.train_sampling_steps), learn_sigma=learn_sigma, pred_sigma=pred_sigma, snr=config.snr_loss 
                            )
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        token_size=config.model_token_size,
                        in_channels=config.in_channels,
                        learn_sigma=learn_sigma,
                        pred_sigma=pred_sigma,
                        **model_kwargs).train()
    logger.info(f"{model.__class__.__name__} Model Parameters: {sum(p.numel() for p in model.parameters()):,}")

    if args.load_from is not None:
        config.load_from = args.load_from
    if config.load_from is not None:
        missing, unexpected = load_checkpoint(
            config.load_from, model, load_ema=config.get('load_ema', False), max_length=max_length)
        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    # prepare for FSDP clip grad norm calculation
    if accelerator.distributed_type == DistributedType.FSDP:
        for m in accelerator._models:
            m.clip_grad_norm_ = types.MethodType(clip_grad_norm_, m)

    # build optimizer and lr scheduler
    lr_scale_ratio = 1
    if config.get('auto_lr', None):
        lr_scale_ratio = auto_scale_lr(config.train_batch_size * get_world_size() * config.gradient_accumulation_steps,
                                       config.optimizer, **config.auto_lr)
    optimizer = build_optimizer(model, config.optimizer)
    lr_scheduler = build_lr_scheduler(config, optimizer, train_dataloader, lr_scale_ratio, accelerator.num_processes)

    timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

    if accelerator.is_main_process:
        tracker_config = dict(vars(config))
        try:
            accelerator.init_trackers(args.tracker_project_name, tracker_config)
        except:
            accelerator.init_trackers(f"tb_{timestamp}")

    start_epoch = 0
    start_step = 0
    skip_step = config.skip_step
    total_steps = len(train_dataloader) * config.num_epochs / accelerator.num_processes

    if config.resume_from is not None and config.resume_from['checkpoint'] is not None:
        resume_path = config.resume_from['checkpoint']
        path = os.path.basename(resume_path)
        start_epoch = int(path.replace('.pth', '').split("_")[1])
        start_step = int(path.replace('.pth', '').split("_")[3]) - 1
        _, missing, unexpected = load_checkpoint(**config.resume_from,
                                                 model=model,
                                                 optimizer=optimizer,
                                                 lr_scheduler=lr_scheduler,
                                                 max_length=max_length,
                                                 )

        logger.warning(f'Missing keys: {missing}')
        logger.warning(f'Unexpected keys: {unexpected}')

    model = accelerator.prepare(model)
    optimizer, train_dataloader, lr_scheduler = accelerator.prepare(optimizer, train_dataloader, lr_scheduler)
    train()
