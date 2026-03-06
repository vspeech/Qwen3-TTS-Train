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
from typing import Any, List, Tuple, Union

import librosa
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from qwen_tts.core.models.configuration_qwen3_tts import Qwen3TTSConfig
from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
from torch.utils.data import Dataset

AudioLike = Union[
    str,                     # wav path, URL, base64
    np.ndarray,              # waveform (requires sr)
    Tuple[np.ndarray, int],  # (waveform, sr)
]

MaybeList = Union[Any, List[Any]]

class TTSDataset(Dataset):
    def __init__(self, data_list, processor, config:Qwen3TTSConfig, lag_num = -1, speaker_field="speaker", dialect_field="dialect"):
        self.data_list = data_list
        self.processor = processor
        self.lag_num = lag_num
        self.config = config
        self.speaker_field = speaker_field
        self.dialect_field = dialect_field

    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str) -> Tuple[np.ndarray, int]:
        
        audio, sr = librosa.load(x, sr=24000, mono=True)

        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)

        return audio.astype(np.float32), int(sr)

    def _normalize_audio_inputs(self, audios: Union[AudioLike, List[AudioLike]]) -> List[Tuple[np.ndarray, int]]:
        """
        Normalize audio inputs into a list of (waveform, sr).

        Supported forms:
          - str: wav path / URL / base64 audio string
          - np.ndarray: waveform (NOT allowed alone here because sr is unknown)
          - (np.ndarray, sr): waveform + sampling rate
          - list of the above

        Args:
            audios:
                Audio input(s).

        Returns:
            List[Tuple[np.ndarray, int]]:
                List of (float32 waveform, original sr).

        Raises:
            ValueError: If a numpy waveform is provided without sr.
        """
        if isinstance(audios, list):
            items = audios
        else:
            items = [audios]

        out: List[Tuple[np.ndarray, int]] = []
        for a in items:
            if isinstance(a, str):
                out.append(self._load_audio_to_np(a))
            elif isinstance(a, tuple) and len(a) == 2 and isinstance(a[0], np.ndarray):
                out.append((a[0].astype(np.float32), int(a[1])))
            elif isinstance(a, np.ndarray):
                raise ValueError("For numpy waveform input, pass a tuple (audio, sr).")
            else:
                raise TypeError(f"Unsupported audio input type: {type(a)}")
        return out

    
    def _build_assistant_text(self, text: str) -> str:
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _build_instruct_text(self, instruct: str) -> str:
        return f"<|im_start|>user\n{instruct}<|im_end|>\n"

    def _ensure_list(self, x: MaybeList) -> List[Any]:
        return x if isinstance(x, list) else [x]
    
    def _tokenize_texts(self, text) -> List[torch.Tensor]:
        input = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        assert sr == 24000, "Only support 24kHz audio"
        mels = mel_spectrogram(
            torch.from_numpy(audio).unsqueeze(0), 
            n_fft=1024, 
            num_mels=128, 
            sampling_rate=24000,
            hop_size=256, 
            win_size=1024, 
            fmin=0, 
            fmax=12000
        ).transpose(1, 2)
        return mels


    def __getitem__(self, idx):
        item = self.data_list[idx]

        audio_path  = item["audio"]
        text        = item["text"]
        audio_codes = item["audio_codes"]
        language        = item.get('language','Auto')
        ref_audio_path  = item['ref_audio']
        
        instruct = item.get('instruct', '')
        
        speaker = item.get(self.speaker_field, 'default_speaker')
        
        dialect = item.get(self.dialect_field, None)

        if instruct:
            full_text = self._build_instruct_text(instruct) + self._build_assistant_text(text)
        else:
            full_text = self._build_assistant_text(text)
            
        text_ids = self._tokenize_texts(full_text)

        audio_codes = torch.tensor(audio_codes, dtype=torch.long)

        ref_audio_list = self._ensure_list(ref_audio_path)
        normalized = self._normalize_audio_inputs(ref_audio_list)
        wav,sr = normalized[0]

        ref_mel = self.extract_mels(audio=wav, sr=sr)

        return {
            "text_ids": text_ids[:,:-5],    # 1 , t
            "audio_codes":audio_codes,      # t, 16
            "ref_mel":ref_mel,
            "speaker": speaker,
            "language": language,
            "dialect": dialect
        }
        
    def collate_fn(self, batch):
        assert self.lag_num == -1

        item_length = [b['text_ids'].shape[1] + b['audio_codes'].shape[0] for b in batch]
        prefix_offset = [9 if b['language'] != 'Auto' else 8 for b in batch]
        item_lengths = [a + b for a, b in zip(item_length, prefix_offset)]
        max_length = max(item_lengths)
        #max_length = max(item_length) + 9
        b,t = len(batch),max_length

        input_ids   = torch.zeros((b,t,2),dtype=torch.long)
        codec_ids   = torch.zeros((b,t,16),dtype=torch.long)
        text_embedding_mask     = torch.zeros((b,t),dtype=torch.bool)
        codec_embedding_mask    = torch.zeros((b,t),dtype=torch.bool)
        codec_mask      = torch.zeros((b,t),dtype=torch.bool)
        attention_mask  = torch.zeros((b,t),dtype=torch.long)
        codec_0_labels  = torch.full((b, t), -100, dtype=torch.long)
        
        speakers = [b['speaker'] for b in batch]
        languages = [b['language'] for b in batch]
        dialects = [b['dialect'] for b in batch]

        for i,data in enumerate(batch):
            text_ids        = data['text_ids']
            audio_codec_0   = data['audio_codes'][:,0]
            audio_codecs    = data['audio_codes']
            language = languages[i]

            text_ids_len = text_ids.shape[1]
            codec_ids_len = audio_codec_0.shape[0]

            if language != 'Auto':
                P = 9 # prefix offset (first text token position)
                num_tts_pad = 5  # positions 4,5,6,7,8
                codec_prefix = torch.tensor([
                    self.config.talker_config.codec_think_id,
                    self.config.talker_config.codec_think_bos_id,
                    0,
                    self.config.talker_config.codec_think_eos_id,
                    0,
                    self.config.talker_config.codec_pad_id,
                ])
                language_mask_pos = 5
            else:
                P = 8
                num_tts_pad = 4
                codec_prefix = torch.tensor([
                    self.config.talker_config.codec_nothink_id,
                    self.config.talker_config.codec_think_bos_id,
                    self.config.talker_config.codec_think_eos_id,
                    0,
                    self.config.talker_config.codec_pad_id,
                ])
                language_mask_pos = None
            
            # text channel
            input_ids[i,  :3, 0] = text_ids[0,:3]
            input_ids[i, 3:3+num_tts_pad, 0] = self.config.tts_pad_token_id
            input_ids[i,   P-1, 0] = self.config.tts_bos_token_id
            input_ids[i, P:P+text_ids_len-3, 0] = text_ids[0,3:]
            input_ids[i,   P+text_ids_len-3, 0] = self.config.tts_eos_token_id
            input_ids[i, P+text_ids_len-2:P+text_ids_len+codec_ids_len , 0] = self.config.tts_pad_token_id

            # codec channel
            # input_ids[i,   :3, 1] = 0
            input_ids[i,    3:3+len(codec_prefix) ,1] = codec_prefix
            input_ids[i,    P:P+text_ids_len-3  ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    P+text_ids_len-3    ,1] = self.config.talker_config.codec_pad_id
            input_ids[i,    P+text_ids_len-2    ,1] = self.config.talker_config.codec_bos_id
            input_ids[i,    P+text_ids_len-1:P+text_ids_len-1+codec_ids_len,    1] = audio_codec_0
            input_ids[i,    P+text_ids_len-1+codec_ids_len,    1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i,    P+text_ids_len-1:P+text_ids_len-1+codec_ids_len] = audio_codec_0
            codec_0_labels[i,    P+text_ids_len-1+codec_ids_len] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, P+text_ids_len-1:P+text_ids_len-1+codec_ids_len,:] = audio_codecs

            text_embedding_mask[i,  :P+text_ids_len+codec_ids_len] = True
            codec_embedding_mask[i, 3:P+text_ids_len+codec_ids_len] = True
            if language_mask_pos is not None:
                codec_embedding_mask[i, language_mask_pos] = False
                codec_embedding_mask[i, 7] = False  
            else:
                codec_embedding_mask[i, 6] = False

            codec_mask[i,   P+text_ids_len-1:P+text_ids_len-1+codec_ids_len] = True
            attention_mask[i, :P+text_ids_len+codec_ids_len] = True
        
        ref_mels = [data['ref_mel'].squeeze(0) for data in batch]
        ref_mel_lengths = torch.tensor([data['ref_mel'].size(1) for data in batch], dtype=torch.int32)
        ref_mels = pad_sequence(ref_mels, batch_first=True, padding_value=0)
        #ref_mels = [data['ref_mel'] for data in batch]
        #ref_mels = torch.cat(ref_mels,dim=0)

        return {
            'input_ids':input_ids,
            'ref_mels':ref_mels,
            'ref_mel_lengths':ref_mel_lengths,
            'attention_mask':attention_mask,
            'text_embedding_mask':text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask':codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels':codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask':codec_mask,
            'speakers': speakers,
            'languages': languages,
            'dialects': dialects
        }
