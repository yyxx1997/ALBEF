export CUDA_VISIBLE_DEVICES=4,5,6,7

# Define hyperparameters
lowlr=(0.0 0.3 0.5 0.7 0.1)
datasets=('f30k')

# Iterate over hyperparameter combinations
for lr in "${lowlr[@]}"; do
  for dataset in "${datasets[@]}"; do
    echo $lr, $dataset
    python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval_entail_lr_split.py \
    --config ./configs/Retrieval_entail_lr_split_${dataset}.yaml \
    --output_dir output/Retrieval_entail_lr_split/${dataset}/${lr} \
    --checkpoint /data1/yx/suda/image-text/data/ALBEF/datas/ALBEF_4M.pth \
    --text_encoder ./output/bert-base-uncased \
    --lowlr $lr
  done
done
