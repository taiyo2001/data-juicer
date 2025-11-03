# ----------------------------------
# Process: Calculate evaluation metrics
# 1. Generate images
# 2. Evaluate generated images
# 3. Calculate final evaluation metrics(No GPU)
# ----------------------------------

# default settings
ICL_NUM=1
ICL_PROMPT="Always prioritize the spatial relationships and object count specified in the prompt."

ICL_NUM=2
ICL_PROMPT="Let's generate step by step"

if [ -n "$ICL_NUM" ] && [ -n "$ICL_PROMPT" ]; then
    MODEL_NAME="${MODEL_NAME}_ICL${ICL_NUM}"
fi
echo "FINAL MODEL_NAME: $MODEL_NAME"

# model name L: max 768 tokens


# ----------------------------------
# 1. Generate images
# ----------------------------------

# ------ SD1.5(77 tokens) ------
MODEL_NAME=SD1_5

# INFO: 'conda activate DetailMaster' is required before.
CUDA_VISIBLE_DEVICES=1 python evaluation_pipeline/image_generation_example/generate_image_sd1_5.py \
--model_name $MODEL_NAME \
--icl_num $ICL_NUM \
--icl_prompt $ICL_PROMPT

# ------ ParaDiffusion(256 tokens) ------
MODEL_NAME=ParaDiffusion
MODEL_NAME=ParaDiffusion_L  # long prompt version(768 tokens)

# INFO: 'conda activate ParaDiffusion' is required before.
CUDA_VISIBLE_DEVICES=0 python evaluation_pipeline/image_generation_example/generate_image_ParaDiffusion.py \
--model_name $MODEL_NAME \
--icl_num $ICL_NUM \
--icl_prompt $ICL_PROMPT

# ------ FLUX(512 tokens) ------
MODEL_NAME=FLUX1-dev
MODEL_NAME=FLUX1-schnell

# INFO: 'conda activate DetailMaster' is required before.
CUDA_VISIBLE_DEVICES=1 python evaluation_pipeline/image_generation_example/generate_image_FLUX1.py \
--model_name $MODEL_NAME \
--icl_num $ICL_NUM \
--icl_prompt $ICL_PROMPT

# ------ LongAlign ------


# ----------------------------------
# 2. Evaluate generated images
# ----------------------------------
# INFO: 'conda activate DetailMaster' is required before.
CUDA_VISIBLE_DEVICES=0 python evaluation_pipeline/eval_process.py \
--image_folder ./evaluation_pipeline/image_generation_example/output_image_$MODEL_NAME \
--image_info_json ./evaluation_pipeline/image_generation_example/output_image_info_$MODEL_NAME.json \
--output_log_dir ./playground/evaluation/$MODEL_NAME \
--output_name_prefix $MODEL_NAME


# ----------------------------------
# 3. Calculate final evaluation metrics(No GPU)
# ----------------------------------
python evaluation_pipeline/cal_eval.py \
--eval_output_log_dir_name ./playground/evaluation/$MODEL_NAME \
--name_prefix $MODEL_NAME

# ----------------------------------
# Other: Results Visualization
# ----------------------------------
# Display generation image
python playground/display_generation_image.py \
--model_name $MODEL_NAME \
--image_id "qual_dev_00003.jpg"

# Generate evaluation comparison table
python playground/eval_comparison_generator.py
