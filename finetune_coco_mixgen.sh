############################### MixGen ###############################
export CUDA_VISIBLE_DEVICES=3,4,5,6
task=Retrieval_coco
sub_task=DataAug

python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
    --config ./configs/${task}.yaml \
    --checkpoint ../Models/ALBEF/ALBEF_4M.pth \
    --text_encoder ../Models/ALBEF/bert-base-uncased \
    --output_dir ../Output/ALBEF/${task}/${sub_task}/re \
    --gradient_accumulation_steps 4 \
    --mode re

mix_mode=('mixgen' 'mixgen_batch' 'mixgen_random')
mix_rate=(0.25 0.15)
mix_lam=(0.5)

# Iterate over hyperparameter combinations

for mr in "${mix_rate[@]}"; do
    for mm in "${mix_mode[@]}"; do
        for ml in "${mix_lam[@]}"; do
            echo $mr, $ml, $mm
            python -m torch.distributed.launch --nproc_per_node=4 --use_env Retrieval.py \
                --config ./configs/${task}.yaml \
                --checkpoint ../Models/ALBEF/ALBEF_4M.pth \
                --text_encoder ../Models/ALBEF/bert-base-uncased \
                --output_dir ../Output/ALBEF/${task}/${sub_task}/${mm}/${mr}_${ml} \
                --gradient_accumulation_steps 4 \
                --mix_rate ${mr}    \
                --mix_lam ${ml} \
                --mode ${mm}
        done
    done
done