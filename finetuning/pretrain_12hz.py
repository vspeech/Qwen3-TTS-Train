# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Qwen3-TTS 12Hz Pre-training Script

This script is designed for pre-training (or continued pre-training) of the
Qwen3-TTS base model. Unlike the SFT script (sft_12hz.py), this script:
  1. Trains ALL parameters (talker + speaker_encoder)
  2. speaker_encoder participates in gradient computation (no .detach())
  3. Saves all weights including speaker_encoder
  4. Uses cosine learning rate scheduler with warmup
  5. Supports gradient checkpointing for memory efficiency
  6. Preserves "base" model type in config
  7. Does not register specific spk_id mappings

Usage:
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
"""

import argparse
import json
import math
import os
import shutil

import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import set_seed
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoConfig


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1):
    """
    Create a cosine annealing schedule with linear warmup.
    After warmup, the learning rate decays from initial lr to min_lr_ratio * initial_lr.
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train():
    parser = argparse.ArgumentParser(description="Qwen3-TTS 12Hz Pre-training Script")
    # Model paths
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                        help="Path to the initial pre-trained model")
    parser.add_argument("--output_model_path", type=str, default="output_pretrain",
                        help="Directory to save checkpoints")
    parser.add_argument("--train_jsonl", type=str, required=True,
                        help="Path to the training data JSONL file")

    # Training hyperparameters
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4, help="Peak learning rate")
    parser.add_argument("--min_lr_ratio", type=float, default=0.1,
                        help="Minimum LR as a fraction of peak LR for cosine schedule")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--max_grad_norm", type=float, default=1.0, help="Max gradient norm for clipping")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")

    # Scheduler
    parser.add_argument("--warmup_steps", type=int, default=1000,
                        help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--warmup_ratio", type=float, default=None,
                        help="Warmup ratio (overrides warmup_steps if set). "
                             "E.g., 0.05 means 5%% of total training steps are warmup.")

    # Memory optimization
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to reduce memory usage")

    # Save strategy
    parser.add_argument("--save_steps", type=int, default=0,
                        help="Save checkpoint every N steps. 0 means only save at end of each epoch.")
    parser.add_argument("--save_total_limit", type=int, default=3,
                        help="Maximum number of checkpoints to keep. Oldest will be deleted. 0 = keep all.")

    # Language params
    parser.add_argument("--default_language", type=str, default="Auto",
                        help="Default language when language field is missing in data")
    parser.add_argument("--language_field", type=str, default="language",
                        help="Field name for language in JSONL")

    # Speaker params
    parser.add_argument("--speaker_field", type=str, default="speaker",
                        help="Field name for speaker in JSONL")
    parser.add_argument("--no_speaker", action="store_true",
                        help="Train without speaker embedding (no ref_audio needed)")

    # Dialect params
    parser.add_argument("--dialect_field", type=str, default="dialect",
                        help="Field name for dialect in JSONL")

    # Sub-talker loss weight
    parser.add_argument("--sub_talker_loss_weight", type=float, default=0.3,
                        help="Weight for the sub-talker (code predictor) loss")

    # Speaker encoder training
    parser.add_argument("--freeze_speaker_encoder", action="store_true",
                        help="Freeze speaker_encoder weights (like SFT mode). "
                             "Default is to train speaker_encoder jointly.")
    parser.add_argument("--speaker_encoder_lr_scale", type=float, default=1.0,
                        help="Learning rate scale factor for speaker_encoder relative to base lr. "
                             "E.g., 0.1 means speaker_encoder uses 10%% of base lr.")

    # Logging
    parser.add_argument("--log_steps", type=int, default=100,
                        help="Log loss every N steps")
    parser.add_argument("--log_with", type=str, default="tensorboard",
                        help="Logging backend (tensorboard, wandb, etc.)")
    parser.add_argument("--project_name", type=str, default="qwen3-tts-pretrain",
                        help="Project name for logging")

    # DataLoader
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")

    args = parser.parse_args()

    # Initialize accelerator
    # Use find_unused_parameters=True because text_projection is not used
    # in the training forward path (only used during inference).
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16",
        log_with=args.log_with,
        project_dir=os.path.join(args.output_model_path, "logs"),
        kwargs_handlers=[ddp_kwargs],
    )

    if args.seed is not None:
        set_seed(args.seed)

    accelerator.print("=" * 60)
    accelerator.print("Qwen3-TTS 12Hz Pre-training")
    accelerator.print("=" * 60)
    accelerator.print(f"Model: {args.init_model_path}")
    accelerator.print(f"Output: {args.output_model_path}")
    accelerator.print(f"Batch size: {args.batch_size}")
    accelerator.print(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    accelerator.print(f"Learning rate: {args.lr}")
    accelerator.print(f"Epochs: {args.num_epochs}")
    accelerator.print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    accelerator.print(f"Freeze speaker encoder: {args.freeze_speaker_encoder}")
    accelerator.print("=" * 60)

    # Load model
    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    # Load training data
    with open(args.train_jsonl) as f:
        train_data = [json.loads(line) for line in f]
    accelerator.print(f"Loaded {len(train_data)} training samples")

    # Check language field
    has_language = any(args.language_field in item for item in train_data)
    if not has_language:
        accelerator.print(f"WARNING: '{args.language_field}' field not found in data, "
                          f"will use default language '{args.default_language}'")

    # Initialize dataset
    dataset = TTSDataset(
        train_data, qwen3tts.processor, config,
        speaker_field=args.speaker_field,
        dialect_field=args.dialect_field,
        no_speaker=args.no_speaker,
    )
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Enable gradient checkpointing
    full_model = qwen3tts.model  # Qwen3TTSForConditionalGeneration (no forward method)
    talker = full_model.talker   # Qwen3TTSTalkerForConditionalGeneration (has forward method)
    speaker_encoder = full_model.speaker_encoder  # Qwen3TTSSpeakerEncoder

    if args.gradient_checkpointing:
        talker.model.gradient_checkpointing = True
        accelerator.print("Gradient checkpointing enabled for talker")

    # Freeze speaker_encoder if requested
    if args.freeze_speaker_encoder and speaker_encoder is not None:
        for param in speaker_encoder.parameters():
            param.requires_grad = False
        accelerator.print("Speaker encoder frozen")

    # Build optimizer with parameter groups
    # Separate speaker_encoder params for potentially different learning rate
    param_groups = []
    talker_params = []
    speaker_encoder_params = []

    for name, param in talker.named_parameters():
        if not param.requires_grad:
            continue
        talker_params.append(param)

    if speaker_encoder is not None:
        for name, param in speaker_encoder.named_parameters():
            if not param.requires_grad:
                continue
            speaker_encoder_params.append(param)

    param_groups.append({
        "params": talker_params,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
    })

    if speaker_encoder_params:
        param_groups.append({
            "params": speaker_encoder_params,
            "lr": args.lr * args.speaker_encoder_lr_scale,
            "weight_decay": args.weight_decay,
        })
        accelerator.print(f"Speaker encoder LR: {args.lr * args.speaker_encoder_lr_scale} "
                          f"(scale={args.speaker_encoder_lr_scale})")

    all_params = list(talker.parameters()) + (list(speaker_encoder.parameters()) if speaker_encoder is not None else [])
    total_trainable = sum(p.numel() for p in all_params if p.requires_grad)
    total_params = sum(p.numel() for p in all_params)
    accelerator.print(f"Trainable parameters: {total_trainable:,} / {total_params:,} "
                      f"({100 * total_trainable / total_params:.1f}%)")

    optimizer = AdamW(param_groups, betas=(0.9, 0.95), eps=1e-8)

    # Calculate total training steps
    num_update_steps_per_epoch = math.ceil(
        len(train_dataloader) / args.gradient_accumulation_steps
    )
    total_training_steps = num_update_steps_per_epoch * args.num_epochs

    # Compute warmup steps
    if args.warmup_ratio is not None:
        warmup_steps = int(total_training_steps * args.warmup_ratio)
    else:
        warmup_steps = args.warmup_steps

    accelerator.print(f"Total training steps: {total_training_steps}")
    accelerator.print(f"Warmup steps: {warmup_steps}")
    accelerator.print(f"Steps per epoch: {num_update_steps_per_epoch}")

    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer, warmup_steps, total_training_steps, args.min_lr_ratio
    )

    code_predictor = talker.code_predictor
    talker.code_predictor = None  # Temporarily detach from talker before DDP wrapping

    if speaker_encoder is not None and not args.freeze_speaker_encoder:
        talker, speaker_encoder, optimizer, train_dataloader, scheduler = accelerator.prepare(
            talker, speaker_encoder, optimizer, train_dataloader, scheduler
        )
    else:
        talker, optimizer, train_dataloader, scheduler = accelerator.prepare(
            talker, optimizer, train_dataloader, scheduler
        )
        if speaker_encoder is not None:
            # Move frozen speaker_encoder to the correct device
            speaker_encoder = speaker_encoder.to(accelerator.device)

    code_predictor = code_predictor.to(accelerator.device)
    raw_talker_before_prepare = accelerator.unwrap_model(talker)
    raw_talker_before_prepare.code_predictor = code_predictor

    # Initialize tracker
    if accelerator.is_main_process:
        accelerator.init_trackers(args.project_name, config=vars(args))

    # Track saved checkpoints for save_total_limit
    saved_checkpoints = []

    # Training loop
    talker.train()
    if speaker_encoder is not None and not args.freeze_speaker_encoder:
        speaker_encoder.train()
    global_step = 0

    # Get unwrapped models for accessing sub-module attributes (embedding layers, etc.)
    raw_talker = accelerator.unwrap_model(talker)
    raw_speaker_encoder = accelerator.unwrap_model(speaker_encoder) if speaker_encoder is not None else None

    for epoch in range(args.num_epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(talker):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                languages_batch = batch.get('languages', [args.default_language] * len(input_ids))

                # --- Speaker Embedding ---
                # In pretraining mode, speaker_encoder participates in gradient computation
                # (no .detach() unlike SFT mode)
                if not args.no_speaker:
                    ref_mels = batch['ref_mels']
                    ref_mel_lengths = batch['ref_mel_lengths']

                    if args.freeze_speaker_encoder:
                        # SFT-like behavior: freeze speaker encoder, use raw to access forward
                        speaker_embedding = raw_speaker_encoder(
                            ref_mels.to(accelerator.device).to(torch.bfloat16),
                            lengths=ref_mel_lengths
                        ).detach()
                    else:
                        # Pretraining: speaker encoder participates in gradient
                        # Use DDP-wrapped speaker_encoder so gradients are synchronized
                        speaker_embedding = speaker_encoder(
                            ref_mels.to(accelerator.device).to(torch.bfloat16),
                            lengths=ref_mel_lengths
                        )

                # --- Build Input Embeddings ---
                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = raw_talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = raw_talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask

                # Inject speaker embedding (position depends on language)
                if not args.no_speaker:
                    for i, language in enumerate(languages_batch):
                        spk_pos = 7 if language.lower() != "auto" else 6
                        input_codec_embedding[i, spk_pos, :] = speaker_embedding[i]

                # Inject language embedding at position 5
                codec_language_id_config = getattr(config.talker_config, 'codec_language_id', None) or {}
                for i, language in enumerate(languages_batch):
                    language_id = None
                    if language.lower() != "auto":
                        if language.lower() not in codec_language_id_config:
                            raise NotImplementedError(
                                f"Language {language} not implemented in codec_language_id: {codec_language_id_config}"
                            )
                        language_id = codec_language_id_config[language.lower()]

                    if language_id is not None:
                        lang_id_tensor = torch.tensor(
                            [language_id], device=input_codec_ids.device, dtype=input_codec_ids.dtype
                        )
                        lang_embedding = raw_talker.model.codec_embedding(lang_id_tensor)
                        input_codec_embedding[i, 5, :] = lang_embedding[0]

                # Combine all embeddings
                input_embeddings = input_text_embedding + input_codec_embedding

                # Detach code_predictor embeddings when building talker input.
                # code_predictor's embedding layers are shared: they are used both here
                # (to build talker input) and later in forward_sub_talker_finetune.
                # If not detached, DDP backward from talker loss would mark these
                # embedding params as ready, and the sub-talker backward would mark
                # them again, causing "marked as ready twice" error.
                # The code_predictor embeddings will still receive proper gradients
                # from forward_sub_talker_finetune.
                for i in range(1, 16):
                    codec_i_embedding = raw_talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding.detach() * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # --- Forward Pass ---
                # Use DDP-wrapped talker for forward so gradients are synchronized across GPUs
                outputs = talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                # Cast to bfloat16 to match code_predictor weights dtype.
                # Under mixed precision, hidden_states may come out as float32
                # from autocast, but small_to_mtp_projection expects bfloat16.
                talker_hidden_states = talker_hidden_states.to(torch.bfloat16)

                # Detach talker_hidden_states from the DDP computation graph.
                # code_predictor is a sub-module of talker (wrapped by DDP together).
                # If we don't detach, backward through forward_sub_talker_finetune
                # will trace back into the DDP graph, causing code_predictor parameters
                # to be marked as "ready" twice — once from the talker forward and once
                # from the sub-talker forward — triggering the DDP error.
                talker_hidden_states = talker_hidden_states.detach()

                sub_talker_logits, sub_talker_loss = raw_talker.forward_sub_talker_finetune(
                    talker_codec_ids, talker_hidden_states
                )

                # --- Backward Pass ---
                # Now that code_predictor is NOT inside DDP (detached before accelerator.prepare),
                # we can safely combine both losses and call accelerator.backward() once.
                # - outputs.loss gradient flows through: talker.model + codec_head (DDP-synced)
                # - sub_talker_loss gradient flows through: code_predictor only (NOT DDP-synced,
                #   each GPU updates independently — acceptable for pretraining)
                # No parameter appears in both paths, so no "marked as ready twice" error.
                loss = outputs.loss + args.sub_talker_loss_weight * sub_talker_loss
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    # Clip gradients for all trainable parameters
                    all_trainable_params = list(talker.parameters())
                    if speaker_encoder is not None and not args.freeze_speaker_encoder:
                        all_trainable_params += list(speaker_encoder.parameters())
                    accelerator.clip_grad_norm_(all_trainable_params, args.max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                combined_loss = outputs.loss.item() + args.sub_talker_loss_weight * sub_talker_loss.item()
                epoch_loss += combined_loss
                epoch_steps += 1

            # Logging
            if accelerator.sync_gradients:
                global_step += 1

                if global_step % args.log_steps == 0:
                    avg_loss = epoch_loss / max(epoch_steps, 1)
                    current_lr = scheduler.get_last_lr()[0]
                    talker_loss_val = outputs.loss.item()
                    sub_loss_val = sub_talker_loss.item()
                    combined_loss_val = talker_loss_val + args.sub_talker_loss_weight * sub_loss_val
                    accelerator.print(
                        f"Epoch {epoch} | Step {global_step} | "
                        f"Loss: {combined_loss_val:.4f} (talker={talker_loss_val:.4f}, sub={sub_loss_val:.4f}) | "
                        f"Avg Loss: {avg_loss:.4f} | LR: {current_lr:.2e}"
                    )
                    if accelerator.is_main_process:
                        accelerator.log({
                            "train/loss": combined_loss_val,
                            "train/talker_loss": talker_loss_val,
                            "train/sub_talker_loss": sub_loss_val,
                            "train/avg_loss": avg_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch,
                            "train/global_step": global_step,
                        }, step=global_step)

                # Save checkpoint at save_steps interval
                if args.save_steps > 0 and global_step % args.save_steps == 0:
                    _save_checkpoint(
                        accelerator, talker, speaker_encoder, config, args,
                        MODEL_PATH, f"checkpoint-step-{global_step}",
                        saved_checkpoints
                    )

        # End of epoch logging
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        accelerator.print(f"Epoch {epoch} completed | Avg Loss: {avg_epoch_loss:.4f}")

        # Save checkpoint at end of each epoch
        _save_checkpoint(
            accelerator, talker, speaker_encoder, config, args,
            MODEL_PATH, f"checkpoint-epoch-{epoch}",
            saved_checkpoints
        )

    # End training
    if accelerator.is_main_process:
        accelerator.end_training()
    accelerator.print("Training completed!")


def _save_checkpoint(accelerator, talker, speaker_encoder, config, args, model_path, ckpt_name, saved_checkpoints):
    """Save a checkpoint with full model weights (including speaker_encoder)."""
    if not accelerator.is_main_process:
        return

    output_dir = os.path.join(args.output_model_path, ckpt_name)
    accelerator.print(f"Saving checkpoint to {output_dir}")

    # Copy model config files
    shutil.copytree(model_path, output_dir, dirs_exist_ok=True)

    # Update config - keep tts_model_type as "base" for pretraining
    config_file = os.path.join(output_dir, "config.json")
    with open(config_file, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    config_dict["tts_model_type"] = "base"

    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)

    # Save model weights (including speaker_encoder)
    # Reconstruct full state_dict with proper key prefixes
    unwrapped_talker = accelerator.unwrap_model(talker)
    state_dict = {}
    for k, v in unwrapped_talker.state_dict().items():
        state_dict[f"talker.{k}"] = v.detach().to("cpu")
    if speaker_encoder is not None:
        unwrapped_spk = accelerator.unwrap_model(speaker_encoder)
        for k, v in unwrapped_spk.state_dict().items():
            state_dict[f"speaker_encoder.{k}"] = v.detach().to("cpu")

    # Save as safetensors - shard if model is large
    total_size = sum(v.numel() * v.element_size() for v in state_dict.values())
    if total_size > 5 * 1024 * 1024 * 1024:  # > 5GB, shard
        _save_sharded(state_dict, output_dir, max_shard_size=4 * 1024 * 1024 * 1024)
    else:
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(state_dict, save_path)

    accelerator.print(f"Checkpoint saved: {output_dir}")

    # Manage save_total_limit
    saved_checkpoints.append(output_dir)
    if args.save_total_limit > 0 and len(saved_checkpoints) > args.save_total_limit:
        oldest = saved_checkpoints.pop(0)
        if os.path.exists(oldest):
            shutil.rmtree(oldest)
            accelerator.print(f"Removed old checkpoint: {oldest}")


def _save_sharded(state_dict, output_dir, max_shard_size=4 * 1024 * 1024 * 1024):
    """Save state_dict as sharded safetensors files."""
    shards = []
    current_shard = {}
    current_size = 0
    weight_map = {}

    for key, tensor in state_dict.items():
        tensor_size = tensor.numel() * tensor.element_size()
        if current_size + tensor_size > max_shard_size and current_shard:
            shards.append(current_shard)
            current_shard = {}
            current_size = 0
        current_shard[key] = tensor
        current_size += tensor_size

    if current_shard:
        shards.append(current_shard)

    total_size = sum(v.numel() * v.element_size() for v in state_dict.values())

    if len(shards) == 1:
        save_path = os.path.join(output_dir, "model.safetensors")
        save_file(shards[0], save_path)
        return

    # Save each shard
    for i, shard in enumerate(shards):
        shard_name = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        save_path = os.path.join(output_dir, shard_name)
        save_file(shard, save_path)
        for key in shard:
            weight_map[key] = shard_name

    # Save index file
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    index_path = os.path.join(output_dir, "model.safetensors.index.json")
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)


if __name__ == "__main__":
    train()
