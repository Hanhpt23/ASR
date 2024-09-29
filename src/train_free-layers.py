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


import torch
import evaluate
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List, Union


from .data.prepare import prepare_dataset, is_audio_in_length_range


logger = logging.getLogger(__name__)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

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

def train_whisper_model(args):

    logging.basicConfig(level=logging.INFO)

    data_raw = load_dataset(args.data_name, name=args.data_subset)

    if args.use_prepare_dataset:
        data = DatasetDict()
        data["train"] = data_raw["train"]
        data["dev"] = data_raw["dev"]
        del data_raw
    else:
        data = data_raw.copy()
        del data_raw

    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model_name)
    normalizer = BasicTextNormalizer()

    if args.freeze_encoder:
        logger.info("Freezing encoder!")
        for name, param in model.named_parameters():
            if name.startswith("model.encoder"):
                param.requires_grad = False

    if not args.freeze_encoder and args.freeze_specific_layers_encoder:
        logger.info(f"Freezing specific layers in encoder: {args.freeze_specific_layers_encoder}")
        for name, param in model.named_parameters():
            if any(f"model.encoder.layers.{layer}." in name for layer in args.freeze_specific_layers_encoder):
                param.requires_grad = False

    if args.freeze_specific_layers_decoder:
        logger.info(f"Freezing specific layers in decoder: {args.freeze_specific_layers_decoder}")
        for name, param in model.named_parameters():
            if any(f"model.decoder.layers.{layer}." in name for layer in args.freeze_specific_layers_decoder):
                param.requires_grad = False
    if not args.language:
        # We only need to set the task id when the language is specified (i.e. in a multilingual setting)
        processor.tokenizer.set_prefix_tokens(language=args.language, task=args.task)
        model.generation_config.language = args.language
        model.generation_config.task = args.task

    if args.cast_audio:
        logger.info("Start casting audio!")
        data = data.cast_column("audio", Audio(sampling_rate=16000))

    if args.use_prepare_dataset:
        logger.info("Start preparing dataset by feature extraction and filtering!")

        logger.info("Starting feature extraction")
        data = data.map(
            lambda examples: prepare_dataset(
                examples, processor, args.do_lower_case, args.do_remove_punctuation
            ), remove_columns=data.column_names["train"], num_proc=args.num_proc)

        logger.info("Starting filtering")
        data["train"] = data["train"].filter(
            is_audio_in_length_range, 
            num_proc=args.num_proc,
            input_columns=["input_length"]
        )

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    def compute_metrics(pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        if args.do_normalize_eval:
            pred_str = [normalizer(pred) for pred in pred_str]
            label_str = [normalizer(label) for label in label_str]

        wer = 100 * wer_metric.compute(predictions=pred_str, references=label_str)
        cer = 100 * cer_metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer, 
                "cer": cer}


    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False

    print(model)

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        gradient_checkpointing=args.gradient_checkpointing,
        evaluation_strategy=args.evaluation_strategy,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        predict_with_generate=args.predict_with_generate,
        generation_max_length=args.generation_max_length,
        num_train_epochs=args.num_train_epochs,
        save_strategy=args.save_strategy,
        logging_steps=args.logging_steps,
        report_to=[args.report_to],
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        push_to_hub=args.push_to_hub,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=data["train"],
        eval_dataset=data["dev"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)

    def log_trainable_parameters(model):
        logger.info("Trainable parameters:")
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(f"{name}")

    log_trainable_parameters(model)


    trainer.train()

    if args.push_to_hub:
        kwargs = {
            "dataset": args.data_name,
            "language": args.language,
            "model_name": args.model_name,
        }
        trainer.push_to_hub(**kwargs)


if __name__ == "__main__":
    args = parse_args()
    pprint(vars(args))
    train_whisper_model(args)