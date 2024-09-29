import json
from tqdm import tqdm
from glob import glob
from datasets import Dataset, DatasetDict, Audio
import argparse

def read_text_to_list(file_path):
    """
    Read text from a file and convert it into a list of JSON objects.

    Args:
        file_path (str): Path to the file.

    Returns:
        list: List of JSON objects.
    """
    with open(file_path, "r") as file:
        data = file.read().replace("]\n", "").replace("[\n", "").replace("\'", "\"")
        data_into_list = data.split(",\n")

    data_jsonl = [json.loads(data_th) for data_th in data_into_list if data_th]
    return data_jsonl

def get_audio_paths(audio_path):
    """
    Get a list of audio file paths from a given directory.

    Args:
        audio_path (str): Path to the directory.

    Returns:
        list: List of audio file paths.
    """
    speakers = glob(f"{audio_path}/*")
    audio_paths = [file for speak_path in speakers for file in glob(f"{speak_path}/*")]
    return audio_paths

def process_data(audio_paths, text_data):
    """
    Process audio and text data to get the final data.

    Args:
        audio_paths (list): List of audio file paths.
        text_data (list): List of JSON objects representing text data.

    Returns:
        tuple: Tuple containing final audio data, final text data, and final duration data.
    """
    audio_path_list, texts_list, durations_list = [], [], []
    for audio_path in tqdm(audio_paths):
        file_name = "/".join(audio_path.split("/")[-2:])
        for data in text_data:
            if data["file"] == file_name:
                audio_path_list.append(audio_path)
                texts_list.append(data["text"])
                durations_list.append(data["duration"])
                break
    return audio_path_list, texts_list, durations_list

def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Process audio and text data.")
    parser.add_argument("--train_audio_path", type=str, help="Path to the audio train directory.")
    parser.add_argument("--test_audio_path", type=str, help="Path to the audio test directory.")
    parser.add_argument("--val_audio_path", type=str, help="Path to the audio test directory.")

    parser.add_argument("--train_scription_path", type=str, help="Path to the train transcript file.")
    parser.add_argument("--test_scription_path", type=str, help="Path to the test transcript file.")
    parser.add_argument("--val_scription_path", type=str, help="Path to the test transcript file.")

    parser.add_argument("--push_to_hub", action=True, help="Enable if want to push dataset to the hub.")
    parser.add_argument("--name_push", type=str, help="Name of the dataset to push to the hub.")

    return parser.parse_args()

if __name__ == "__main__":
    # Usage example
    args = parse_args()
    text_train = read_text_to_list(file_path=args.train_audio_path)
    text_test = read_text_to_list(file_path=args.test_scription_path)
   
    data_audio_train_paths = get_audio_paths(args.train_audio_path)
    data_audio_test_paths = get_audio_paths(args.test_audio_path)    

    data_audio_train, text_train, durations_train = process_data(data_audio_train_paths, text_train)
    data_audio_test, text_test, durations_test = process_data(data_audio_test_paths, text_test)
    
    # if args.push_to_hub:
    #     data = DatasetDict()
    #     data["train"] = Dataset.from_dict({"audio": data_audio_train, "text": text_train, "duration": durations_train}).cast_column("audio", Audio())
    #     data["test"] = Dataset.from_dict({"audio": data_audio_test, "text": text_test, "duration": durations_test}).cast_column("audio", Audio())
    #     data["val"] = Dataset.from_dict({"audio": data_audio_test, "text": text_test, "duration": durations_test}).cast_column("audio", Audio())
        
    #     data.push_to_hub(args.name_push, private=True)

