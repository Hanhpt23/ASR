from datasets import load_dataset, DatasetDict, Audio
from transformers import WhisperProcessor
from transformers.models.whisper.english_normalizer import BasicTextNormalizer
import argparse

normalizer = BasicTextNormalizer()


def prepare_dataset(batch, processor, do_lower_case=None, do_remove_punctuation=None):
    # load and (possibly) resample audio data to 16kHz
    audio = batch["audio"]

    # compute log-Mel input features from input audio array 
    batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
    # compute input length of audio sample in seconds
    batch["input_length"] = len(audio["array"]) / audio["sampling_rate"]
    
    # optional pre-processing steps
    transcription = batch["text"]
    if do_lower_case:
        transcription = transcription.lower()
    if do_remove_punctuation:
        transcription = normalizer(transcription).strip()
    
    # encode target text to label ids
    batch["labels"] = processor.tokenizer(transcription).input_ids
    return batch


def is_audio_in_length_range(length, max_input_length=30.0):
    return length < max_input_length



def str2bool(v):
    if isinstance(v, bool):
       return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def argparser():
    parser = argparse.ArgumentParser(description="Prepare dataset for Whisper training")
    parser.add_argument("--model_name", type=str, default=None, help="Pretrained model name")
    parser.add_argument("--data_name", type=str, default=None, help="Dataset name")
    parser.add_argument("--data_subset", type=str, default="default", help="Dataset subset")
    parser.add_argument("--language", type=str, default="de", help="Language for the dataset")
    parser.add_argument("--do_lower_case", type=str2bool, help="Whether to convert text to lowercase")
    parser.add_argument("--do_remove_punctuation", type=str2bool, help="Whether to remove punctuation from text")
    parser.add_argument("--name_dataset_output", type=str, default="prepared_dataset", help="Output dataset name")
    parser.add_argument("--num_proc", type=int, default=1, help="Number of processes for parallel processing")
    return parser.parse_args()


if __name__ == "__main__":
    args = argparser()
    # load dataset
    data_raw = load_dataset(args.data_name, name=args.data_subset)

    data = DatasetDict()
    data["train"] = data_raw["train"]
    data["dev"] = data_raw["dev"]

    # initialize processor
    processor = WhisperProcessor.from_pretrained(args.model_name, language=args.language, task="transcribe")

    data = data.cast_column("audio", Audio(sampling_rate=16000))

    data = data.map(
        lambda examples: prepare_dataset(
            examples, processor, args.do_lower_case, args.do_remove_punctuation
        ), remove_columns=data.column_names["train"], num_proc=args.num_proc)

    data["train"] = data["train"].filter(
        is_audio_in_length_range, 
        num_proc=args.num_proc,
        input_columns=["input_length"]
    )

    # save dataset
    data.push_to_hub(args.name_dataset_output)