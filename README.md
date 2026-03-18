# Qwen3-TTS Quick Start Guide

## Overview

This guide provides a complete quick start walkthrough for Qwen3-TTS training, an all-in-one solution from data preparation to model training.

## 1. Environment Setup

### 1.1 Install Dependencies
```bash
git clone https://github.com/vspeech/Qwen3-TTS-Train.git
cd Qwen3-TTS-Train/finetuning

# Install required Python packages
pip install librosa numpy tqdm transformers accelerate safetensors
```

### 1.2 Check Environment
```bash
# Check Python environment
python --version

# Check CUDA availability
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 2. Data Preparation Quick Start

### 2.1 Input JSONL format

Prepare your training file as a JSONL (one JSON object per line). Each line must contain:

- `audio`: path to the target training audio (wav)
- `text`: transcript corresponding to `audio`
- `ref_audio`: path to the reference speaker audio (wav)
- `speaker`: "spk1" (multi-speaker required)
- `language`: "Chinese" (multi-language required)
- `instruct`: "用特别愤怒的语气说" (instruct required)

### 2.2 Data Preparation

Convert `train_raw.jsonl` into a training JSONL that includes `audio_codes`:

```bash
# Prepare data
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

## 3. Training Quick Start

### 3.1 Single-Speaker Training

```bash
# Single-speaker training
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --speaker_name speaker_test
```

### 3.2 Multi-Speaker Training

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/utt0001.wav","speaker":"spk1"}
{"audio":"./data/utt0002.wav","text":"他要大家仔细听听湖边群山送过来的回音。","ref_audio":"./data/utt0002.wav","speaker":"spk2"}
```

```bash
# Multi-speaker training
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path multi_speaker_output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --multi_speaker 
```

### 3.3 Multi-Language Training

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/utt0001.wav","speaker":"spk1","language":"Chinese"}
{"audio":"./data/utt0002.wav","text":"She said she would be here by noon.","ref_audio":"./data/utt0002.wav","speaker":"spk2","language":"English"}
```

```bash
# Multi-language training
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path multi_language_output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --multi_speaker \
  --multi_language
```

### 3.3 Instruct Training

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/utt0001.wav","instruct":"用特别愤怒的语气说"}
```

```bash
# Instruct training
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path instruct_output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --instruct_model
```

### 3.4 No Speaker and No Language Training

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。"}
{"audio":"./data/utt0002.wav","text":"他要大家仔细听听湖边群山送过来的回音。"}
```

```bash
python sft_12hz.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --output_model_path instruct_output \
  --train_jsonl train_with_codes.jsonl \
  --batch_size 32 \
  --lr 2e-6 \
  --num_epochs 10 \
  --no_speaker
```

### 3.5 Continued Pre-Training

Example:
```jsonl
{"audio":"./data/utt0001.wav","text":"其实我真的有发现，我是一个特别善于观察别人情绪的人。","ref_audio":"./data/spk1.wav","speaker":"spk1","language":"Chinese"}
{"audio":"./data/utt0002.wav","text":"他要大家仔细听听湖边群山送过来的回音。","ref_audio":"./data/spk2.wav","speaker":"spk2","language":"Chinese"}
```

```bash
accelerate launch pretrain_12hz.py \
    --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --train_jsonl /path/to/large_scale_data.jsonl \
    --output_model_path /path/to/output \
    --batch_size 4 \
    --lr 1e-4 \
    --num_epochs 10 \
    --warmup_steps 1000 \
    --gradient_checkpointing \
    --save_steps 5000
```

## 4. Best Practices Summary

### 4.1 Data Preparation
- ✅ Ensure audio quality (SNR > 20dB)
- ✅ Audio duration of 2-10 seconds is recommended
- ✅ At least 1 hour of data for single speaker
- ✅ At least 30 minutes per speaker for multi-speaker

### 4.2 Training Configuration
- ✅ Single speaker: batch_size=2, lr=2e-6, epochs=3
- ✅ Multi-speaker: enable --multi_speaker flag
- ✅ Small dataset: increase epochs, reduce batch_size

### 4.3 Performance Optimization
- ✅ Use GPU for training
- ✅ Enable mixed precision
- ✅ Set batch_size appropriately
- ✅ Use SSD storage for faster data loading

**Happy training!** 🎉

With this quick start guide, you should be able to quickly get started with the Qwen3-TTS training pipeline. If you have any questions, please refer to the detailed documentation or try adjusting parameters.

## Acknowledgement
This repo is directly based on the following excellent project:
 
- [**Qwen3-TTS**](https://github.com/QwenLM/Qwen3-TTS)
