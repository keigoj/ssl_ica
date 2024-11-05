from torch.utils.data import Dataset
import json
import os
import torch
import torchaudio
import torchaudio.functional as F
import re
import pandas as pd
from typing import Union, List


class DatasetFromManifest(Dataset):
    def remove_special_characters(self, text):
        chars_to_remove_regex = [",", "?", "¿", ".", "!", "¡", ";", "；", ":", '""', "%", '"', "�", "ʿ", "·", "჻", "~", "՞",
                                "؟", "،", "।", "॥", "«", "»", "„", "“", "”", "「", "」", "‘", "’", "《", "》", "(", ")", "[", "]",
                                "{", "}", "=", "`", "_", "+", "<", ">", "…", "–", "°", "´", "ʾ", "‹", "›", "©", "®", "—", "→", "。",
                                "、", "﹂", "﹁", "‧", "～", "﹏", "，", "｛", "｝", "（", "）", "［", "］", "【", "】", "‥", "〽", "$", "#"
                                "『", "』", "〝", "〟", "⟨", "⟩", "〜", "：", "！", "？", "♪", "؛", "/", "\\", "º", "−", "^", "'", "ʻ", "ˆ", " ", "・", ]
        chars_to_remove_regex = "".join(chars_to_remove_regex)
        chars_to_ignore_regex = f"[{re.escape(''.join(chars_to_remove_regex))}]"
        text = re.sub(chars_to_ignore_regex, '', text).lower()
        return text
    
    def __init__(self, manifest: Union[str, List[str]]) -> None:
        self.data_list = []
        print(f"loadding {manifest}")
        self.data = []
        if isinstance(manifest, list):  # manifestがリストの場合
            for m in manifest:
                self.load_manifest(m)
        elif isinstance(manifest, str):  # manifestが単一のファイルパスの場合
            self.load_manifest(manifest)
        else:
            raise ValueError("manifest should be a list of file paths or a single file path.")
        
    def load_manifest(self, manifest_path: str) -> None:
        with open(manifest_path, "r") as file:
            for line in file:
                json_data = json.loads(line)
                text = json_data["text"]
                if text == "":
                    continue
                self.data.append({
                    "path": json_data["audio_filepath"],
                    "text": text
                })
    
    def __getitem__(self, idx: int):
        waveform, sr  = torchaudio.load(self.data[idx]["path"])
        waveform = F.resample(waveform=waveform, orig_freq=sr, new_freq=16000)
        return waveform[0], self.data[idx]["text"]

    def __len__(self) -> int:
        return len(self.data)
    
    def gather_all_label(self):
        sentences = [[c for c in data["text"]] for data in self.data]
        sentences = ["".join(l) for l in sentences]
        text = "".join(sentences)
        labels = sorted(list(set(text)))
        labels = dict(zip(labels, range(len(labels))))
        return labels
    
    def output_hf_csv(self, path):
        df = pd.DataFrame(self.data)
        #path->audio, transcription->sentenceに変更
        # df = df.rename(columns={'path': 'audio', 'transcription': 'sentence'})
        df.to_csv(path, index=False)