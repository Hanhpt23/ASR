python3 -m src.train \
    --output_dir "./whisper-vietmed-v2" \
    --model_name "openai/whisper-tiny" \
    --data_name "pphuc25/VietMed-train-test" \
    --data_subset "default" \
    --cast_audio True \
    --use_prepare_dataset True \
    --do_lower_case False \
    --do_remove_punctuation False \
    --max_input_length 100 \
    --do_normalize_eval True \
    --language vi \
    --learning_rate 0.0001 \
    --warmup_steps 50 \
    --freeze_encoder True \
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