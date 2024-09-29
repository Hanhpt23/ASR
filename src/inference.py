import argparse
import logging
import sys
from pprint import pprint
from datasets import load_dataset, Audio, DatasetDict
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer
    )
import json
from transformers.pipelines.pt_utils import KeyDataset
import torch
import evaluate
from transformers import pipeline
from dataclasses import dataclass
from typing import Any, Dict, List, Union


from .data.prepare import prepare_dataset, is_audio_in_length_range


logger = logging.getLogger(__name__)


def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():

    parser = argparse.ArgumentParser(description='Train a Whisper model.')

    parser.add_argument('--data_name', type=str, help='Name of the data to load.')
    parser.add_argument('--model_name', type=str, help='name of the model to load.')
    parser.add_argument('--inference_model_name', type=str, help='name of the model to load.')
    parser.add_argument('--data_subset', type=str, help='Subset of the data to use.', default=None)
    parser.add_argument('--cast_audio', type=str2bool, default=None, help='Whether to cast audio data.')
    parser.add_argument('--use_prepare_dataset', type=str2bool, default=None, help='Whether to use prepared dataset.')
    parser.add_argument('--do_lower_case', type=str2bool, default=None, help='Whether to convert text to lower case.')
    parser.add_argument('--do_remove_punctuation', type=str2bool, default=None, help='Whether to remove punctuation from text.')
    parser.add_argument('--max_input_length', type=int, default=None, help='Maximum input length for audio.')
    parser.add_argument('--do_normalize_eval', type=str2bool, default=None, help='Whether to normalize evaluation.')
    parser.add_argument('--num_proc', type=int, default=1, help='Num proc to multiprocess.')
    parser.add_argument('--language', type=str, default=None, help='Language of the data.')
    parser.add_argument('--task', type=str, default='transcribe', help='Task of the data.')

    # Training arguments
    parser.add_argument('--output_dir', type=str, default='./', help='Output directory for the model.')
    parser.add_argument('--per_device_train_batch_size', type=int, default=8, help='Training batch size per device.')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Number of gradient accumulation steps.')
    parser.add_argument('--learning_rate', type=float, default=1e-5, help='Learning rate.')
    parser.add_argument('--warmup_steps', type=int, default=500, help='Number of warmup steps.')
    parser.add_argument('--gradient_checkpointing', type=str2bool, default=True, help='Whether to use gradient checkpointing.')
    parser.add_argument('--evaluation_strategy', type=str, default='epoch', help='Evaluation strategy.')
    parser.add_argument('--per_device_eval_batch_size', type=int, default=8, help='Evaluation batch size per device.')
    parser.add_argument('--predict_with_generate', type=str2bool, default=True, help='Whether to predict with generation.')
    parser.add_argument('--generation_max_length', type=int, default=225, help='Maximum generation length.')
    parser.add_argument('--num_train_epochs', type=int, default=None, help='Number of training epochs.')
    parser.add_argument('--save_strategy', type=str, default='epoch', help='Save strategy.')
    parser.add_argument('--logging_steps', type=int, default=25, help='Number of steps between logging.')
    parser.add_argument('--report_to', type=str, default='tensorboard', help='Where to report results.')
    parser.add_argument('--load_best_model_at_end', type=str2bool, default=True, help='Whether to load the best model at the end.')
    parser.add_argument('--metric_for_best_model', type=str, default='wer', help='Metric for best model.')
    parser.add_argument('--greater_is_better', type=str2bool, default=False, help='Whether greater metric is better.')
    parser.add_argument('--push_to_hub', type=str2bool, default=True, help='Whether to push the model to the hub.')

    parser.add_argument('--freeze_encoder', type=str2bool, default=None, help='Whether to freeze the encoder.')
    parser.add_argument('--freeze_specific_layers_encoder', type=str, default=None, help='Specific layers to freeze in the encoder.')
    parser.add_argument('--freeze_specific_layers_decoder', type=str, default=None, help='Specific layers to freeze in the decoder.')

    args = parser.parse_args()

    if args.freeze_specific_layers_encoder:
        args.freeze_specific_layers_encoder = list(map(int, args.freeze_specific_layers_encoder.replace(",", " ").split()))
    if args.freeze_specific_layers_decoder:
        args.freeze_specific_layers_decoder = list(map(int, args.freeze_specific_layers_decoder.replace(",", " ").split()))

    return args

def inference_whisper_model(args):

    logging.basicConfig(level=logging.INFO)

    data_raw = load_dataset(args.data_name, name=args.data_subset)

    if args.use_prepare_dataset:
        data = DatasetDict()
        data["train"] = data_raw["train"]
        data["dev"] = data_raw["dev"]
        data["test"] = data_raw["test"]
        del data_raw
    else:
        data = data_raw.copy()
        del data_raw
    # print(data)
    print(f"Keys in dataset: {data['dev'].features.keys()}")

    if args.cast_audio:
        logger.info("Start casting audio!")
        data = data.cast_column("audio", Audio(sampling_rate=16000))

    inference(model_name=args.inference_model_name, dataset=data["dev"],
              save_name='evaluation', start_idx =0, end_idx=50)
    
    inference(model_name=args.inference_model_name, dataset=data["test"], 
              save_name='test', start_idx =0, end_idx=50)

def inference(model_name, dataset, save_name, start_idx, end_idx):
    pipe = pipeline(model=model_name, device=0)
    results = []
    # selected_data = dataset.select(range(start_idx, end_idx))
    selected_data = dataset
    # for i, out in enumerate(pipe(KeyDataset(dataset.select(range(start_idx, end_idx)), "audio"))):
    for i, out in enumerate(pipe(KeyDataset(dataset, "audio"))):
        result = {
            # "sample": start_idx + i,
            "sample": i,
            "prediction": out,
            "ground_truth": selected_data[i]["text"]
        }
        results.append(result)
        # print(result)
        # break
    with open(f"inference_{save_name}.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    args = parse_args()
    pprint(vars(args))
    inference_whisper_model(args)



