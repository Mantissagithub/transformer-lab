# Meeting Summarization Training Notebook

## Overview

This Jupyter notebook (`meeting_summarization_training.ipynb`) consolidates all Python files from the transformer project into a single, executable notebook for training a Transformer model on the MeetingBank dataset.

## What's Included

The notebook contains all the code from these files:
- `config.py` - Configuration settings
- `layer_normalization.py` - Layer normalization component
- `residual_connection.py` - Residual connections
- `input_embedding.py` - Input embeddings
- `positional_encoding.py` - Positional encoding
- `multihead_attention.py` - Multi-head attention mechanism
- `feed_forward.py` - Feed-forward networks
- `projection_layer.py` - Output projection layer
- `encoder_block.py` - Encoder blocks and encoder
- `decoder_block.py` - Decoder blocks and decoder
- `transformer_model.py` - Complete transformer model
- `dataset.py` - Dataset classes for MeetingBank
- `train.py` - Training loop and data loading

## Prerequisites

```bash
pip install torch torchvision torchaudio
pip install datasets transformers tokenizers
pip install tqdm tensorboard
```

## How to Use

### Option 1: Run All Cells
1. Open `meeting_summarization_training.ipynb` in Jupyter or VS Code
2. Run all cells sequentially (Kernel → Restart & Run All)
3. Training will start automatically

### Option 2: Step-by-Step Execution
1. Run cells 1-11: Set up model architecture and functions
2. Run cell 12: Download dataset and train tokenizers (takes time)
3. Run cell 13: Initialize model and training components
4. Run cell 15: Start training

### Adjusting Configuration

Before running cell 15 (training), you can modify the configuration in cell 2:

```python
config = get_config()
config['num_epochs'] = 5      # Reduce for faster testing
config['batch_size'] = 4      # Reduce if out of memory
config['max_src_len'] = 1024  # Reduce for faster processing
```

## Dataset: MeetingBank

- **Source**: https://huggingface.co/datasets/huuuyeah/meetingbank
- **Size**: 6,892 segment-level summarization instances
- **Task**: Generate summaries from meeting transcripts
- **Download**: Automatic on first run (~2-3 GB)

## Training Configuration

Default settings:
- **Batch size**: 8
- **Epochs**: 20
- **Learning rate**: 0.0001
- **Max source length**: 2048 tokens (transcript)
- **Max target length**: 512 tokens (summary)
- **Model**: 6 encoder layers, 6 decoder layers, 8 heads

## Output Files

### Model Checkpoints
- `weights/meeting_model_00.pt` - Epoch 0 checkpoint
- `weights/meeting_model_01.pt` - Epoch 1 checkpoint
- ... (one per epoch)

### Tokenizers
- `tokenizers/meetingbank_transcript.json` - Source tokenizer
- `tokenizers/meetingbank_summary.json` - Target tokenizer

### Logs
- `runs/meeting_summarization/` - TensorBoard logs
- `meetingbank_dataset.json` - Sample data (first 10 items)

## Monitoring Training

### TensorBoard
Open a terminal and run:
```bash
tensorboard --logdir runs/meeting_summarization
```
Then navigate to http://localhost:6006

### Progress Bar
The notebook displays a progress bar with:
- Current epoch
- Batch progress
- Current loss value

## Testing After Training

After training completes, test the model:

```python
# Run the inference test
test_sample_inference(model, val_dataloader, src_tokenizer, tgt_tokenizer, device)
```

This will:
1. Take a sample from the validation set
2. Generate a summary using the trained model
3. Display the source transcript, actual summary, and generated summary

## Memory Requirements

### GPU Training (Recommended)
- **Minimum**: 8 GB VRAM
- **Recommended**: 16 GB VRAM
- **Training time**: 2-4 hours per epoch (depends on GPU)

### CPU Training
- **Minimum**: 16 GB RAM
- **Recommended**: 32 GB RAM
- **Training time**: 10-20 hours per epoch (very slow)

### Reducing Memory Usage
If you encounter out-of-memory errors:

```python
config['batch_size'] = 2          # Reduce batch size
config['max_src_len'] = 1024      # Reduce max sequence length
config['max_tgt_len'] = 256       # Reduce max target length
```

## Troubleshooting

### Issue: Out of Memory
**Solution**: Reduce `batch_size`, `max_src_len`, or `max_tgt_len` in config

### Issue: Dataset Download Fails
**Solution**: Check internet connection and try again. Dataset is ~2-3 GB

### Issue: Tokenizer Training Takes Too Long
**Solution**: First run trains tokenizers (10-15 min). Subsequent runs load cached tokenizers

### Issue: CUDA Not Available
**Solution**: Training will use CPU automatically (much slower)

## File Structure After Running

```
transformer/
├── meeting_summarization_training.ipynb  # Main notebook
├── weights/                               # Model checkpoints
│   ├── meeting_model_00.pt
│   ├── meeting_model_01.pt
│   └── ...
├── tokenizers/                           # Trained tokenizers
│   ├── meetingbank_transcript.json
│   └── meetingbank_summary.json
├── runs/                                 # TensorBoard logs
│   └── meeting_summarization/
├── meetingbank_dataset.json              # Sample data
└── [original .py files remain unchanged]
```

## Citation

If you use this code or the MeetingBank dataset, please cite:

```bibtex
@inproceedings{hu-etal-2023-meetingbank,
    title = "MeetingBank: A Benchmark Dataset for Meeting Summarization",
    author = "Yebowen Hu and Tim Ganter and Hanieh Deilamsalehy and Franck Dernoncourt and Hassan Foroosh and Fei Liu",
    booktitle = "Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (ACL)",
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
}
```

## Support

For issues or questions:
1. Check this README
2. Review the markdown cells in the notebook
3. Check the MeetingBank dataset page: https://huggingface.co/datasets/huuuyeah/meetingbank

## Notes

- First run will download the dataset (~2-3 GB) and train tokenizers (~10-15 min)
- Subsequent runs will use cached dataset and tokenizers
- Training is resource-intensive; consider using a GPU
- You can stop training anytime (Kernel → Interrupt) and resume from the last checkpoint
