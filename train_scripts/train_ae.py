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
from accelerate.utils import DistributedType, DistributedDataParallelKwargs
from mmcv.runner import LogBuffer

from diffusion.data.builder import build_dataset, build_datawrapper, set_data_root
from diffusion.model.builder import build_model
from diffusion.utils.checkpoint import save_checkpoint, load_checkpoint
from diffusion.utils.dist_utils import synchronize, get_world_size, clip_grad_norm_, flush
from diffusion.utils.logger import get_root_logger, rename_file_with_creation_time
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.misc import set_random_seed, read_config, init_random_seed, DebugUnderflowOverflow
from diffusion.utils.optimizer import build_optimizer, auto_scale_lr
from diffusion.model.garment_losses import build_loss


def set_fsdp_env():
    os.environ["ACCELERATE_USE_FSDP"] = 'true'
    os.environ["FSDP_AUTO_WRAP_POLICY"] = 'TRANSFORMER_BASED_WRAP'
    os.environ["FSDP_BACKWARD_PREFETCH"] = 'BACKWARD_PRE'

def log_validation(model, step, save_folder):
    torch.cuda.empty_cache()
    model = accelerator.unwrap_model(model).eval()

    global valid_dataloader_iter

    try:
        # Get the next batch from the DataLoader
        batch = next(valid_dataloader_iter)
    except StopIteration:
        # Restart the iterator if it reaches the end
        valid_dataloader_iter = iter(valid_dataloader)
        batch = next(valid_dataloader_iter)

    
    max_edge_num = config.start_config['max_panel_len']
    max_pattern_len = config.start_config['max_pattern_len']
    cloths_info = batch['ground_truth']
    bs = len(batch['data_folder'])
    pattern = cloths_info['outlines']
    stitch_tags = cloths_info['stitch_tags']
    rots = cloths_info['rotations'].unsqueeze(-2).repeat(1, 1, max_edge_num, 1)
    tranls = cloths_info['translations'].unsqueeze(-2).repeat(1, 1, max_edge_num, 1)
    gt_seq = torch.concat([pattern, stitch_tags, rots, tranls], dim=-1)

    edge_type = cloths_info['edge_type'] # bs, max_panel_num, max_edge_num, 4
    edge_mask = (~cloths_info['empty_edges_mask']).int().unsqueeze(-1) # bs, max_panel_num, max_edge_num, 1
    stitch_per_edge = (~cloths_info['free_edges_mask']).int().unsqueeze(-1) # bs, max_panel_num, max_edge_num, 1
    reverse_stitch = cloths_info['reverse_stitch'].int()
    discrete_quantization = torch.concat([edge_type, stitch_per_edge, reverse_stitch, edge_mask], dim=-1) # bs, max_panel_num, max_edge_num, 10
    discrete_quantization = discrete_quantization * 2 - 1
    input_data = torch.concat([gt_seq, discrete_quantization], dim=-1)
    input_data = einops.rearrange(input_data, 'b p e c -> b (p e) c')
    
    recovered_cloth, _, _ = model(input_data)
    
    recovered_cloth = recovered_cloth.reshape(bs, max_pattern_len, max_edge_num, -1)
    discrete = recovered_cloth[..., -10:]
    continuous = recovered_cloth[..., :-10]
    recovered_cloths_info = {
        'rotations': continuous[..., -7:-3].mean(-2), # max_pattern_len, 4
        'translations': continuous[..., -3:].mean(-2), # max_pattern_len, 3
        'outlines': continuous[..., :9], # max_pattern_len, max_edge_num, 9-element_size
        'stitch_tags': continuous[..., 9:12], # max_pattern_len, max_edge_num, 3
        'edge_mask' : discrete[..., -1], # max_pattern_len, max_edge_num
        'reverse_stitch': discrete[..., -2].unsqueeze(-1), # max_pattern_len, max_edge_num
        'free_edges_mask': -discrete[..., -3], # max_pattern_len, max_edge_num
        'edge_type': discrete[..., :-3], # max_pattern_len, max_edge_num, 4
    }

    torch.cuda.empty_cache()
    folder_for_preds = os.path.join(save_folder+'_'+str(step))
    try:
        batch_img_files = datawrapper.dataset.save_prediction_batch(
                    recovered_cloths_info, 
                    batch['name'], batch['data_folder'],
                    save_to=folder_for_preds)
    except Exception as e:
        logger.warning(f"Error in {step} saving validation: {e}")

    # flush()
    return 


def train():
    if config.get('debug_nan', False):
        DebugUnderflowOverflow(model)
        logger.info('NaN debugger registered. Start to detect overflow during training.')
    time_start, last_tic = time.time(), time.time()
    log_buffer = LogBuffer()

    global_step = start_step + 1
    max_panel = config.data.start_config['max_pattern_len']
    max_edge = config.data.start_config['max_panel_len']

    # Now you train the model
    for epoch in range(start_epoch + 1, config.num_epochs + 1):
        data_time_start= time.time()
        data_time_all = 0
        for step, batch in enumerate(train_dataloader):
            if step < skip_step:
                global_step += 1
                continue    # skip data in the resumed ckpt
            cloths_info = batch['ground_truth']
            bs = len(batch['data_folder'])
            pattern = cloths_info['outlines']
            stitch_tags = cloths_info['stitch_tags']
            rots = cloths_info['rotations'].unsqueeze(-2).repeat(1, 1, max_edge, 1)
            tranls = cloths_info['translations'].unsqueeze(-2).repeat(1, 1, max_edge, 1)
            gt_seq = torch.concat([pattern, stitch_tags, rots, tranls], dim=-1)

            edge_type = cloths_info['edge_type'] # bs, max_panel_num, max_edge_num, 4
            edge_mask = (~cloths_info['empty_edges_mask']).int().unsqueeze(-1) # bs, max_panel_num, max_edge_num, 1
            stitch_per_edge = (~cloths_info['free_edges_mask']).int().unsqueeze(-1) # bs, max_panel_num, max_edge_num, 1
            reverse_stitch = cloths_info['reverse_stitch'].int()
            discrete_quantization = torch.concat([edge_type, stitch_per_edge, reverse_stitch, edge_mask], dim=-1) # bs, max_panel_num, max_edge_num, 10
            discrete_quantization = discrete_quantization * 2 - 1
            input_data = torch.concat([gt_seq, discrete_quantization], dim=-1)
            input_data = einops.rearrange(input_data, 'b p e c -> b (p e) c')

            # Sample a random timestep for each image
            grad_norm = None
            data_time_all += time.time() - data_time_start
            if step % config.quality_eval == 0:
                criterion.with_quality_eval()
            with accelerator.accumulate(model):
                optimizer.zero_grad()
                recovered_cloth, _, e_latent_loss = model(input_data)
                recovered_cloth = recovered_cloth.reshape(bs, max_panel, max_edge, -1)
                discrete = recovered_cloth[..., -10:]
                continuous = recovered_cloth[..., :-10]
                recovered_cloths_info = {
                    'rotations': continuous[..., -7:-3].mean(-2), # max_pattern_len, 4
                    'translations': continuous[..., -3:].mean(-2), # max_pattern_len, 3
                    'outlines': continuous[..., :9], # max_pattern_len, max_edge_num, 9-element_size
                    'stitch_tags': continuous[..., 9:12], # max_pattern_len, max_edge_num, 3
                    'edge_mask' : discrete[..., -1], # max_pattern_len, max_edge_num
                    'reverse_stitch': discrete[..., -2].unsqueeze(-1), # max_pattern_len, max_edge_num
                    'free_edges_mask': -discrete[..., -3], # max_pattern_len, max_edge_num
                    'edge_type': discrete[..., :-3], # max_pattern_len, max_edge_num, 4
                }
                cloth_loss = criterion(recovered_cloths_info, cloths_info, epoch=epoch)
                loss = cloth_loss[0]*config.tokenizer_config['loss_keypoint']['cloth_loss_w']
                loss += e_latent_loss * config.tokenizer_config['loss_keypoint']['e_loss_w']
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                lr_scheduler.step()
            criterion.without_quality_eval()

            lr = lr_scheduler.get_last_lr()[0]
            logs = {args.loss_report_name: accelerator.gather(loss).mean().item()}
            logs['reconstruct_loss'] = accelerator.gather(e_latent_loss).mean().item()
            for key in cloth_loss[1].keys():
                if cloth_loss[1][key] is not None:
                    cloth_loss[1][key] = torch.tensor(cloth_loss[1][key]).to(accelerator.device)
                    try:
                        logs[key] = accelerator.gather(cloth_loss[1][key]).mean().item()
                    except:
                        logs[key] = accelerator.gather(cloth_loss[1][key])
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
                    with torch.no_grad():
                        log_validation(model, global_step, save_folder=os.path.join(config.work_dir, 'visualize'))

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
        default="train-compression",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
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
    # ddp_handler = DistributedDataParallelKwargs(find_unused_parameters=False)
    # Initialize accelerator and tensorboard logging
    if config.use_fsdp:
        init_train = 'FSDP'
        from accelerate import FullyShardedDataParallelPlugin
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullStateDictConfig
        set_fsdp_env()
        fsdp_plugin = FullyShardedDataParallelPlugin(state_dict_config=FullStateDictConfig(offload_to_cpu=False, rank0_only=False))
    else:
        init_train = 'DDP'
        fsdp_plugin = None

    even_batches = True

    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with=args.report_to,
        project_dir=os.path.join(config.work_dir, "logs"),
        fsdp_plugin=fsdp_plugin,
        even_batches=even_batches,
        kwargs_handlers=[init_handler],
    )

    log_name = 'train_log.log'
    if accelerator.is_main_process:
        if os.path.exists(os.path.join(config.work_dir, log_name)):
            rename_file_with_creation_time(os.path.join(config.work_dir, log_name))
    logger = get_root_logger(os.path.join(config.work_dir, log_name), name='Tokenzier')

    logger.info(accelerator.state)
    config.seed = init_random_seed(config.get('seed', None))
    set_random_seed(config.seed)

    if accelerator.is_main_process:
        config.dump(os.path.join(config.work_dir, 'config.py'))

    logger.info(f"Config: \n{config.pretty_text}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.seed}")
    logger.info(f"Initializing: {init_train} for training")
    image_size = config.image_size
    max_length = config.model_max_length

    # build dataloader
    set_data_root(config.data_root)
    dataset = build_dataset(
        config.data, resolution=image_size, aspect_ratio_type=config.aspect_ratio_type,
        real_prompt_ratio=config.real_prompt_ratio, max_length=max_length, config=config,
    )
    known_split = {'filename': config.data.known_split}
    datawrapper = build_datawrapper(config.datawrapper, dataset, num_workers=config.num_workers, known_split=known_split, batch_size=config.train_batch_size, validation_size=config.validation_size, shuffle=True)
    train_dataloader = datawrapper.get_loader('train')
    valid_dataloader = datawrapper.get_loader('validation')

    # build models
    criterion = build_loss(config, train=True)
    model = build_model(config.model,
                        config.grad_checkpointing,
                        config.get('fp32_attention', False),
                        input_token_num=config.token_size,
                        tokenizer=config.tokenizer_config,
                        input_channels=config.continous_channels+config.discrete_channels).train()
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
    lr_scale_ratio = 5
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
    valid_dataloader = accelerator.prepare(valid_dataloader)
    valid_dataloader_iter = iter(valid_dataloader)
    train()
