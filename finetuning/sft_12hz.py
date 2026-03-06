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
import argparse
import json
import os
import shutil

import torch
from accelerate import Accelerator
from dataset import TTSDataset
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig

target_speaker_embedding = None
target_speaker_embedding_sum = None
target_speaker_embedding_count = 0
speaker_embeddings = {}
speaker_embeddings_sum = {}
speaker_embeddings_count = {}


def train():
    global target_speaker_embedding, speaker_embeddings
    global target_speaker_embedding_sum, target_speaker_embedding_count
    global speaker_embeddings_sum, speaker_embeddings_count

    parser = argparse.ArgumentParser()
    parser.add_argument("--init_model_path", type=str, default="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    parser.add_argument("--output_model_path", type=str, default="output")
    parser.add_argument("--train_jsonl", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--speaker_name", type=str, default="speaker_test")
    # multi speaker params 
    parser.add_argument("--multi_speaker", action="store_true", help="Enable multi-speaker training mode")
    parser.add_argument("--speaker_field", type=str, default="speaker", help="Field name for speaker in JSONL")
    parser.add_argument("--max_speakers", type=int, default=1000, help="Maximum number of supported speakers")
    parser.add_argument("--start_spk_id", type=int, default=3000, help="Starting index for speaker spk_id")
    # instruct params
    parser.add_argument("--instruct_model", action="store_true", help="Whether to use instruct training mode")
    parser.add_argument("--instruct_field", type=str, default="instruct", help="Field name for instruction in JSONL")
    # language params
    parser.add_argument("--multi_language", action="store_true", help="Enable multi-language training mode")
    parser.add_argument("--language_field", type=str, default="language", help="Field name for language in JSONL")
    parser.add_argument("--default_language", type=str, default="Auto", help="Default language when the language field is not present in data")
    parser.add_argument("--start_lang_id", type=int, default=2075, help="Starting index for language lang_id")
    # dialect params
    parser.add_argument("--dialect", type=str, default=None,
                        help="Single dialect name for all speakers, e.g. 'cantonese_dialect'. "
                             "Mutually exclusive with --multi_dialect.")
    parser.add_argument("--dialect_token_id", type=int, default=None,
                        help="Token ID for the single dialect in codec_embedding. "
                             "If not set, will auto-assign from start_lang_id.")
    parser.add_argument("--multi_dialect", action="store_true",
                        help="Enable multi-dialect training mode. Each JSONL record must have a 'dialect' field "
                             "(e.g. 'cantonese_dialect', 'sichuan_dialect'). Each dialect will get a unique token_id. "
                             "Each speaker will be associated with the dialect from its data.")
    parser.add_argument("--dialect_field", type=str, default="dialect",
                        help="Field name for dialect in JSONL (used when --multi_dialect is set)")
    parser.add_argument("--start_dialect_id", type=int, default=None,
                        help="Starting token ID for multi-dialect assignment. "
                             "Defaults to start_lang_id if not set.")
    args = parser.parse_args()

    accelerator = Accelerator(gradient_accumulation_steps=4, mixed_precision="bf16", log_with="tensorboard")

    MODEL_PATH = args.init_model_path

    qwen3tts = Qwen3TTSModel.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    config = AutoConfig.from_pretrained(MODEL_PATH)

    train_data = open(args.train_jsonl).readlines()
    train_data = [json.loads(line) for line in train_data]
    
    # Check if instruct field exists
    has_instruct = any(args.instruct_field in item for item in train_data)
    if args.instruct_model and not has_instruct:
        print(f"WARNING: Instruct training mode enabled, but '{args.instruct_field}' field not found in data")
    
    # Check if language field exists
    has_language = any(args.language_field in item for item in train_data)
    if not has_language:
        print(f"WARNING: '{args.language_field}' field not found in data, will use default language '{args.default_language}'")

    # Validate dialect params
    if args.dialect is not None and args.multi_dialect:
        raise ValueError("--dialect and --multi_dialect are mutually exclusive. Use --dialect for single dialect, --multi_dialect for multiple dialects.")

    # dialect_id_mapping: dialect_name -> token_id  (built during validation, used in training & saving)
    dialect_id_mapping = {}
    # spk_dialect_map: speaker_name -> dialect_name  (built from data, used for config saving)
    spk_dialect_map = {}

    if args.dialect is not None:
        # Single dialect mode
        if "dialect" not in args.dialect:
            print(f"WARNING: Dialect name '{args.dialect}' does not contain 'dialect'. Adding '_dialect' suffix.")
            args.dialect = args.dialect + "_dialect"
        if args.dialect_token_id is None:
            args.dialect_token_id = args.start_lang_id
            print(f"Auto-assigned dialect_token_id = {args.dialect_token_id} (from start_lang_id)")
        dialect_id_mapping[args.dialect] = args.dialect_token_id
        print(f"Single dialect training: name='{args.dialect}', token_id={args.dialect_token_id}")

    if args.multi_dialect:
        # Multi-dialect mode: scan data to discover all dialects
        if args.start_dialect_id is None:
            args.start_dialect_id = args.start_lang_id
        dialects_set = set()
        for item in train_data:
            if args.dialect_field in item:
                d = item[args.dialect_field]
                if d:
                    dialects_set.add(d)
        if not dialects_set:
            raise ValueError(f"--multi_dialect enabled but no '{args.dialect_field}' field found in data")
        dialects_list = sorted(dialects_set)
        for i, d in enumerate(dialects_list):
            name = d if "dialect" in d else d + "_dialect"
            dialect_id_mapping[name] = args.start_dialect_id + i
        print(f"Multi-dialect training enabled. Discovered {len(dialects_list)} dialects:")
        for name, tid in dialect_id_mapping.items():
            print(f"  {name} -> token_id {tid}")

        # Build speaker -> dialect mapping from data
        for item in train_data:
            spk = item.get(args.speaker_field, args.speaker_name)
            d = item.get(args.dialect_field, None)
            if d:
                d_key = d if "dialect" in d else d + "_dialect"
                if spk in spk_dialect_map and spk_dialect_map[spk] != d_key:
                    print(f"WARNING: Speaker '{spk}' has multiple dialects: '{spk_dialect_map[spk]}' and '{d_key}'. Using latest.")
                spk_dialect_map[spk] = d_key
    
    # Multi-speaker training logic
    speakers = []
    languages = []
    if args.multi_speaker or args.multi_language:
        # Extract all speakers from data
        speakers_set = set()
        languages_set = set()
        for item in train_data:
            if args.speaker_field in item:
                speakers_set.add(item[args.speaker_field])
            if args.language_field in item:
                languages_set.add(item[args.language_field])

        if len(speakers_set) == 0:
            print(f"WARNING: Multi-speaker training mode enabled, but '{args.speaker_field}' field not found in data")
            args.multi_speaker = False
        else:
            speakers = list(speakers_set)
            print(f"Detected {len(speakers)} speakers: {speakers}")
            if len(speakers) > args.max_speakers:
                print(f"WARNING: Number of speakers ({len(speakers)}) exceeds maximum limit ({args.max_speakers}), using first {args.max_speakers} speakers")
                speakers = speakers[:args.max_speakers]
        if len(languages_set) == 0:
            print(f"WARNING: Multi-language training mode enabled, but '{args.language_field}' field not found in data")
            args.multi_language = False
        else:
            languages = list(languages_set)
            print(f"Detected {len(languages)} languages: {languages}")

    
    # Initialize dataset, pass speaker_field and dialect_field parameters
    dataset = TTSDataset(train_data, qwen3tts.processor, config, speaker_field=args.speaker_field, dialect_field=args.dialect_field)
    train_dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=dataset.collate_fn)

    optimizer = AdamW(qwen3tts.model.parameters(), lr=args.lr, weight_decay=0.01)

    model, optimizer, train_dataloader = accelerator.prepare(
        qwen3tts.model, optimizer, train_dataloader
    )

    num_epochs = args.num_epochs
    model.train()

    for epoch in range(num_epochs):
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):

                input_ids = batch['input_ids']
                codec_ids = batch['codec_ids']
                ref_mels = batch['ref_mels']
                ref_mel_lengths = batch['ref_mel_lengths']
                text_embedding_mask = batch['text_embedding_mask']
                codec_embedding_mask = batch['codec_embedding_mask']
                attention_mask = batch['attention_mask']
                codec_0_labels = batch['codec_0_labels']
                codec_mask = batch['codec_mask']
                speakers_batch = batch.get('speakers', [args.speaker_name] * len(input_ids))
                languages_batch = batch.get('languages', [args.default_language] * len(input_ids))
 

                #speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype)).detach()
                speaker_embedding = model.speaker_encoder(ref_mels.to(model.device).to(model.dtype), lengths=ref_mel_lengths).detach()

                # Multi-speaker training: accumulate embeddings for averaging
                if args.multi_speaker and speakers:
                    for i, speaker in enumerate(speakers_batch):
                        emb = speaker_embedding[i:i+1].detach()
                        if speaker not in speaker_embeddings_sum:
                            speaker_embeddings_sum[speaker] = emb.clone()
                            speaker_embeddings_count[speaker] = 1
                        else:
                            speaker_embeddings_sum[speaker] += emb
                            speaker_embeddings_count[speaker] += 1
                        # Keep latest for compatibility (will use average when saving)
                        speaker_embeddings[speaker] = emb
                else:
                    # Single-speaker training: accumulate for averaging
                    emb = speaker_embedding.mean(dim=0, keepdim=True).detach()
                    if target_speaker_embedding_sum is None:
                        target_speaker_embedding_sum = emb.clone()
                        target_speaker_embedding_count = 1
                    else:
                        target_speaker_embedding_sum += emb
                        target_speaker_embedding_count += 1
                    target_speaker_embedding = emb  # keep latest for compatibility

                input_text_ids = input_ids[:, :, 0]
                input_codec_ids = input_ids[:, :, 1]

                input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
                input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
                input_codec_embedding[:, 7, :] = speaker_embedding
                # Inject language/dialect embedding at position 5
                # This matches modeling_qwen3_tts.py lines 2135-2148:
                #   codec_prefill_list = [[think_id, think_bos_id, language_id, think_eos_id]]
                #   → language_id is at index 2, which maps to position 5 in global coords
                codec_language_id_config = getattr(config.talker_config, 'codec_language_id', None) or {}
                dialects_batch = batch.get('dialects', [None] * len(input_ids))
                for i, language in enumerate(languages_batch):
                    language_id = None
                    # Step 1: Resolve base language_id from language field
                    if language.lower() != "auto":
                        if language.lower() not in codec_language_id_config:
                            raise NotImplementedError(f"Language {language} not implemented in codec_language_id: {codec_language_id_config}")
                        language_id = codec_language_id_config[language.lower()]

                    # Step 2: Dialect override (takes priority over language)
                    if args.multi_dialect:
                        # Multi-dialect: use per-sample dialect field
                        sample_dialect = dialects_batch[i]
                        if sample_dialect:
                            d_key = sample_dialect if "dialect" in sample_dialect else sample_dialect + "_dialect"
                            if d_key in dialect_id_mapping:
                                language_id = dialect_id_mapping[d_key]
                            else:
                                raise ValueError(f"Dialect '{d_key}' not found in dialect_id_mapping: {dialect_id_mapping}")
                    elif args.dialect is not None:
                        # Single dialect: all samples use the same dialect token_id
                        language_id = args.dialect_token_id
                    else:
                        # No dialect training: check pre-existing spk_is_dialect config
                        speaker = speakers_batch[i]
                        spk_is_dialect = getattr(config.talker_config, 'spk_is_dialect', None) or {}
                        if (language.lower() in ["chinese", "auto"] and
                                speaker is not None and speaker != "" and
                                speaker.lower() in spk_is_dialect and
                                spk_is_dialect[speaker.lower()] != False):
                            dialect = spk_is_dialect[speaker.lower()]
                            if dialect in codec_language_id_config:
                                language_id = codec_language_id_config[dialect]

                    if language_id is not None:
                        lang_id_tensor = torch.tensor([language_id], device=input_codec_ids.device, dtype=input_codec_ids.dtype)
                        lang_embedding = model.talker.model.codec_embedding(lang_id_tensor)
                        input_codec_embedding[i, 5, :] = lang_embedding[0]
                

                input_embeddings = input_text_embedding + input_codec_embedding

                for i in range(1, 16):
                    codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
                    codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
                    input_embeddings = input_embeddings + codec_i_embedding

                # Use correct model call method (languages parameter not supported)
                outputs = model.talker(
                    inputs_embeds=input_embeddings[:, :-1, :],
                    attention_mask=attention_mask[:, :-1],
                    labels=codec_0_labels[:, 1:],
                    output_hidden_states=True
                )

                hidden_states = outputs.hidden_states[0][-1]
                talker_hidden_states = hidden_states[codec_mask[:, 1:]]
                talker_codec_ids = codec_ids[codec_mask]

                sub_talker_logits, sub_talker_loss = model.talker.forward_sub_talker_finetune(talker_codec_ids, talker_hidden_states)

                loss = outputs.loss + 0.3 * sub_talker_loss

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)

                optimizer.step()
                optimizer.zero_grad()

            if step % 100 == 0:
                accelerator.print(f"Epoch {epoch} | Step {step} | Loss: {loss.item():.4f}")

        if accelerator.is_main_process:
            output_dir = os.path.join(args.output_model_path, f"checkpoint-epoch-{epoch}")
            shutil.copytree(MODEL_PATH, output_dir, dirs_exist_ok=True)

            input_config_file = os.path.join(MODEL_PATH, "config.json")
            output_config_file = os.path.join(output_dir, "config.json")
            with open(input_config_file, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            
            # Set model type based on whether instruct training is used
            if args.instruct_model:
                config_dict["tts_model_type"] = "custom_voice"
            else:
                config_dict["tts_model_type"] = "custom_voice"

            talker_config = config_dict.get("talker_config", {})
            
            # Multi-speaker config logic
            if args.multi_speaker and speakers:
                spk_id_mapping = {}
                spk_dialect_mapping = {}
                for i, speaker in enumerate(speakers):
                    spk_id = args.start_spk_id + i
                    spk_id_mapping[speaker] = spk_id
                    # Determine dialect for this speaker
                    if args.multi_dialect:
                        # Use per-speaker dialect from data
                        spk_dialect_mapping[speaker] = spk_dialect_map.get(speaker, False)
                    elif args.dialect is not None:
                        # Single dialect: all speakers share the same dialect
                        spk_dialect_mapping[speaker] = args.dialect
                    else:
                        spk_dialect_mapping[speaker] = False
                
                talker_config["spk_id"].update(spk_id_mapping)
                talker_config["spk_is_dialect"].update(spk_dialect_mapping)
            else:
                talker_config["spk_id"].update({
                    args.speaker_name: args.start_spk_id
                })
                if args.multi_dialect:
                    talker_config["spk_is_dialect"].update({
                        args.speaker_name: spk_dialect_map.get(args.speaker_name, False)
                    })
                elif args.dialect is not None:
                    talker_config["spk_is_dialect"].update({
                        args.speaker_name: args.dialect
                    })
                else:
                    talker_config["spk_is_dialect"].update({
                        args.speaker_name: False
                    })
            
            if args.multi_language and languages:
                codec_language_mapping = {}
                for i, language in enumerate(languages):
                    lang_id = args.start_lang_id + i
                    codec_language_mapping[language.lower()] = lang_id
                
                talker_config["codec_language_id"].update(codec_language_mapping)
            
            # Write dialect -> token_id mappings to codec_language_id
            if dialect_id_mapping:
                talker_config["codec_language_id"].update(dialect_id_mapping)
            elif args.dialect is None and not args.multi_dialect and not (args.multi_language and languages):
                talker_config["codec_language_id"].update({
                    args.default_language.lower(): args.start_lang_id
                })
            
            config_dict["talker_config"] = talker_config

            with open(output_config_file, 'w', encoding='utf-8') as f:
                json.dump(config_dict, f, indent=2, ensure_ascii=False)

            unwrapped_model = accelerator.unwrap_model(model)
            state_dict = {k: v.detach().to("cpu") for k, v in unwrapped_model.state_dict().items()}

            drop_prefix = "speaker_encoder"
            keys_to_drop = [k for k in state_dict.keys() if k.startswith(drop_prefix)]
            for k in keys_to_drop:
                del state_dict[k]

            weight = state_dict['talker.model.codec_embedding.weight']
            
            # Multi-speaker embedding storage logic: use average embedding
            if args.multi_speaker and speakers:
                for i, speaker in enumerate(speakers):
                    spk_id = args.start_spk_id + i
                    if speaker in speaker_embeddings_sum:
                        avg_emb = speaker_embeddings_sum[speaker] / speaker_embeddings_count[speaker]
                        state_dict['talker.model.codec_embedding.weight'][spk_id] = avg_emb[0].to(weight.device).to(weight.dtype)
                        accelerator.print(f"Speaker '{speaker}' (spk_id={spk_id}): saved average embedding from {speaker_embeddings_count[speaker]} samples")
            else:
                # Single-speaker embedding storage: use average embedding
                if target_speaker_embedding_sum is not None and target_speaker_embedding_count > 0:
                    avg_emb = target_speaker_embedding_sum / target_speaker_embedding_count
                    state_dict['talker.model.codec_embedding.weight'][args.start_spk_id] = avg_emb[0].to(weight.device).to(weight.dtype)
                    accelerator.print(f"Single speaker (spk_id={args.start_spk_id}): saved average embedding from {target_speaker_embedding_count} batches")
            
            save_path = os.path.join(output_dir, "model.safetensors")
            save_file(state_dict, save_path)

if __name__ == "__main__":
    train()
