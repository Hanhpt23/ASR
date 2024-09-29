#!/bin/bash

#SBATCH --job-name=S       # create a short name for your job
#SBATCH --output=./gpu_output/Test_small_encode-vietmed-JOB_ID_%j-%N.log # create output file

#SBATCH --nodes=1                  # node count
#SBATCH --ntasks-per-node=1        #CPUS per node (how many cpu to use withinin 1 node)
#SBATCH --mem=250G
#SBATCH --time=100:00:00               # total run time limit (HH:MM:SS)
#SBATCH --partition=gpu1 --gres=gpu  # number of gpus per node

echo "Job ID: $SLURM_JOBID"
echo "Node names: $SLURM_JOB_NODELIST"
echo "VietMed: Free encoder - Fall, wer"
python3 -m src.train \
    --output_dir "./whisper-small-Encod-vietmed" \
    --model_name "openai/whisper-small" \
    --data_name "pphuc25/VietMed-split-8-2" \
    --data_subset "default" \
    --cast_audio True \
    --use_prepare_dataset True \
    --do_lower_case True \
    --do_remove_punctuation True \
    --max_input_length 100 \
    --do_normalize_eval True \
    --language vi \
    --learning_rate 0.0001 \
    --warmup_steps 50 \
    --freeze_encoder False \
    --gradient_checkpointing True \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 4 \
    --save_strategy "epoch" \
    --evaluation_strategy "epoch" \
    --predict_with_generate True \
    --generation_max_length 225 \
    --logging_steps 25 \
    --report_to "wandb" \
    --load_best_model_at_end True \
    --metric_for_best_model "wer" \
    --greater_is_better False \
    --push_to_hub True