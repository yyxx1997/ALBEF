############################### MixGen ###############################
export CUDA_VISIBLE_DEVICES=3,4,5,6
task=Retrieval_coco
sub_task=DataAug
mode=mixgen_clip_score

mix_rate=(0.25 0.1 0.5)
mix_lam=(0.5)

# Iterate over hyperparameter combinations
for mr in "${mix_rate[@]}"; do
    for ml in "${mix_lam[@]}"; do
        echo $mr, $ml, $mode
        python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
            --config ./configs/${task}_cscore.yaml \
            --checkpoint ../Models/ALBEF/ALBEF_4M.pth \
            --text_encoder ../Models/ALBEF/bert-base-uncased \
            --output_dir ../Output/ALBEF/${task}/${sub_task}/$mode/${mr}_${ml} \
            --gradient_accumulation_steps 4 \
            --mix_rate ${mr}    \
            --mix_lam ${ml} \
            --mode $mode
    done
done