CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=8678 \
      train_scripts/train_fisrt_step.py \
      configs/diffusion/diffusion_stage_one.py \
      --work-dir output/diffusion_stage_one

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 --master_port=8678 \
      train_scripts/train_second_step.py \
      configs/diffusion/diffusion_stage_two.py \
      --work-dir output/diffusion_stage_two