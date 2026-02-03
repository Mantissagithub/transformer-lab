from pathlib import Path

def get_config():
    return {
        "batch_size": 8,
        "num_epochs": 20,
        "lr": 10**-4,
        "seq_len": 512,
        "max_src_len": 512,
        "max_tgt_len": 256,
        "dataset_name": "huuuyeah/meetingbank",
        "model_folder" : "weights",
        "model_basename": "meeting_model_mhc",
        "preload": None,
        "tokenizer_file": "tokenizers/meetingbank_{}.json",
        "experiment_name": "meeting_summarization_mhc",
        "additional_param": "value",
        "vocab_size": 32000,
        "task": "summarization",
        # Hyper Connection parameters
        "use_hyper_connection": True,  # Set to True to use HyperConnections instead of residual connections
        "hyper_n": 4,  # Width of hyper connection (number of previous layer outputs to use)
        # Transformer architecture parameters
        "d_model": 512,  # Model dimension
        "d_ff": 2048,  # Feed-forward dimension
        "h": 8,  # Number of attention heads
        "n_layers": 6,  # Number of encoder/decoder layers
        "dropout": 0.1  # Dropout rate
    }

def get_weights_file_path(config, epoch: str):
    model_folder = Path(config['model_folder'])
    model_basename = config['model_basename']
    model_filename = model_folder / f"{model_basename}{epoch}.pt"
    return str(Path('.')/model_folder/model_filename)
