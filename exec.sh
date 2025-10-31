# default settings
ICL_NUM=1
ICL_PROMPT="Always prioritize the spatial relationships and object count specified in the prompt."

ICL_NUM=2
ICL_PROMPT="Let's generate step by step"

if [ -n "$ICL_NUM" ] && [ -n "$ICL_PROMPT" ]; then
    MODEL_NAME="${MODEL_NAME}_ICL${ICL_NUM}"
fi
echo "FINAL MODEL_NAME: $MODEL_NAME"

# ------ SD1.5 ------
# generate
# NOTE: 'conda activate DetailMaster' is required before. Exec from data-juicer/evaluation_pipeline/image_generation_example/
CUDA_VISIBLE_DEVICES=1 python generate_image_sd1_5.py \
--model_path stable-diffusion-v1-5/stable-diffusion-v1-5 \
--prompt_path ../../DetailMaster_Dataset/DetailMaster_Dataset.json \
--output_json ./output_image_info.json
# evaluate
# NOTE: 'conda activate DetailMaster' is required before
CUDA_VISIBLE_DEVICES=1 python evaluation_pipeline/eval_process.py \
--image_folder ./evaluation_pipeline/image_generation_example/output_image \
--image_info_json ./evaluation_pipeline/image_generation_example/output_image_info.json \
--output_log_dir ./playground/evaluation \
--output_name_prefix SD1_5
# calculate final eval metrics(No GPU)
python evaluation_pipeline/cal_eval.py \
--eval_output_log_dir_name ./playground/evaluation \
--name_prefix SD1_5


# ------ ParaDiffusion ------
MODEL_NAME=ParaDiffusion

# generate
# NOTE: 'conda activate $MODEL_NAME' is required before
CUDA_VISIBLE_DEVICES=1 python evaluation_pipeline/image_generation_example/generate_image_ParaDiffusion.py \
--model_name $MODEL_NAME \
--icl_num $ICL_NUM \
--icl_prompt $ICL_PROMPT
# evaluate
# NOTE: 'conda activate DetailMaster' is required before
CUDA_VISIBLE_DEVICES=0 python evaluation_pipeline/eval_process.py \
--image_folder ./evaluation_pipeline/image_generation_example/output_image_$MODEL_NAME \
--image_info_json ./evaluation_pipeline/image_generation_example/output_image_info_$MODEL_NAME.json \
--output_log_dir ./playground/evaluation/$MODEL_NAME \
--output_name_prefix $MODEL_NAME
# calculate final eval metrics(No GPU)
python evaluation_pipeline/cal_eval.py \
--eval_output_log_dir_name ./playground/evaluation/$MODEL_NAME \
--name_prefix $MODEL_NAME


# ------ FLUX ------
MODEL_NAME=FLUX1-dev
MODEL_NAME=FLUX1-schnell

# generate
# NOTE: 'conda activate DetailMaster' is required before
CUDA_VISIBLE_DEVICES=0 python evaluation_pipeline/image_generation_example/generate_image_FLUX1.py \
--model_name $MODEL_NAME \
--icl_num $ICL_NUM \
--icl_prompt $ICL_PROMPT


# ------ LongAlign ------
