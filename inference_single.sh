device_num="0"
num_ddim_steps=50
tau=25
beta=0.8
gamma=0.2
category=cat
input_image=cat_1
task="cat2cat wearing glasses"

CUDA_VISIBLE_DEVICES=${device_num} python src/edit_once.py \
    --input_image "sample_data/${category}/${input_image}.png" \
    --task_name "${task}" \
    --results_folder "output" \
    --num_ddim_steps "${num_ddim_steps}" \
    --negative_guidance_scale 5.0 \
    --tau "${tau}" \
    --beta "${beta}" \
    --gamma "${gamma}" \
    --use_float_16
