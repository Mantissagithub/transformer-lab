from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 512,
        "max_src_len": 2048,
        "max_tgt_len": 512,
        "dataset_name": "huuuyeah/meetingbank",
        "model_folder" : "weights",
        "model_basename": "meeting_model_",
        "preload": None,
        "tokenizer_file": "tokenizers/meetingbank_{}.json",
        "experiment_name": "meeting_summarization",
        "additional_param": "value",
        "vocab_size": 32000,
        "task": "summarization"  
    }

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder'])
    model_basename = config['model_basename']
    model_filename = model_folder / f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
