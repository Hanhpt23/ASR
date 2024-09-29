python3 -m src.data.prepare \
    --model_name "openai/whisper-tiny" \
    --data_name "pphuc25/VietMed-split-8-2" \
    --data_subset "default" \
    --language "vi" \
    --do_lower_case False \
    --do_remove_punctuation False \
    --name_dataset_output "pphuc25/VietMed-prepared" \
    --num_proc 1