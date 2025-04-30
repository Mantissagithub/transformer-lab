from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 350,
        "lang_src": "en",
        "lang_tgt": "ta",
        "model_folder" : "weights",
        "model_basename": "tmodel_",
        "preload": None,
        "tokenizer_file": "tokenizers/{}.json",
        "experiment_name": "transformer_translation",
        "additional_param": "value",
        "vocab_size": 30000
    }

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder'])
    model_basename = config['model_basename']
    model_filename = model_folder / f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
