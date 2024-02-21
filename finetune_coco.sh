export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
python -m torch.distributed.launch --nproc_per_node=8 --use_env Retrieval.py \
  --config ./configs/Retrieval_coco.yaml \
  --output_dir ../Output/ALBEF/Retrieval_coco \
  --checkpoint ../Models/ALBEF/ALBEF_4M.pth \
  --text_encoder ../Models/ALBEF/bert-base-uncased \
  --gradient_accumulation_steps 2 \
  --eval_before_train
