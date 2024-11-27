from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor
from transformers import get_linear_schedule_with_warmup
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config
from wav2vec2_att import Wav2Vec2ForCTCWeighted
from g2p_en import G2p
import torch
from dataloader import MyDataLoader
from dataset_from_manifest import DatasetFromManifest
import torchaudio
import logging


phn_vocab_path = "./data/vocab_phoneme.json"
vocab_path = "./data/vocab_subword.json"

# feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
# tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="<unk>", pad_token="<blank>", word_delimiter_token="▁")
# phn_processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)

# path = "/home/hojo/espnet_ctc/egs2/librispeech_100/asr1/downloads/LibriSpeech/dev-clean/6295/64301/6295-64301-0000.flac"
# path1 = "/home/hojo/espnet_ctc/egs2/librispeech_100/asr1/downloads/LibriSpeech/dev-clean/6295/64301/6295-64301-0001.flac"

# x, sr = torchaudio.load(path)
# x1, sr1 = torchaudio.load(path1)
# wav = torch.cat([x, x1], dim=0)
# print(wav.shape)


# g2p = G2p()
# text = "BUT IN HIS HANDS SOLITUDE AND A VIOLIN WERE SURE TO MARRY IN MUSIC"
# g2p_list = g2p(text)
# g2p_list = ["|" if phn == " " else phn for phn in g2p_list]
# phoneme = "".join(g2p_list)

# a = phn_processor.tokenizer(g2p_list, return_tensors="pt", padding=True).input_ids
# a_flat = a.view(-1) 
# b = phn_processor.tokenizer(phoneme, return_tensors="pt", padding=True).input_ids

# print(g2p_list, len(g2p_list))
# print(phoneme, len(phoneme))
# # print(a_flat, a_flat.shape)
# print(b, b.shape)



train_dataset_path = ["data/train-clean-100-manifest.json"]
valid_dataset_path = ["data/dev-manifest.json"]
test_dataset_path =  ["data/test-clean-manifest.json", "data/test-other-manifest.json"]

# data_module = MyDataLoader(
#     train_dataset_path, 
#     valid_dataset_path, 
#     test_dataset_path, 
#     batch_size=32,
#     ngpu=1
# )

# train_dataset_path = "data/train-clean-100-manifest.json"
# train_dataset = DatasetFromManifest(train_dataset_path)
# n_samples = len(train_dataset)
# train_sub, _ = torch.utils.data.random_split(train_dataset, [n_samples//10, n_samples-n_samples//10])

# print(n_samples)
# print(len(train_sub))
# print(type(train_sub))
# print(train_sub)

logger = logging.getLogger("test")
logging.info("finished successfully")


# Some weights of Wav2Vec2ForCTCWeighted were not initialized from the model checkpoint at /home/hojo/exp/ssl2/model/base_lr1e-4_warm300_30epoch_2/checkpoint-11596 and are newly initialized: ['multihead_attn_low.linear_k.weight', 'conditioning_layer_phn.weight', 'multihead_attn_upp.linear_out.bias', 'multihead_attn_upp.linear_v.bias', 'multihead_attn_upp.linear_k.weight', 'multihead_attn_upp.linear_out.weight', 'multihead_attn_upp.linear_k.bias', 'multihead_attn_upp.linear_v.weight', 'multihead_attn_low.linear_q.weight', 'multihead_attn_low.linear_out.bias', 'multihead_attn_low.linear_q.bias', 'multihead_attn_upp.linear_q.bias'
# , 'conditioning_layer.bias', 'multihead_attn_low.linear_out.weight', 'lm_haed_phn.weight', 'multihead_attn_low.linear_v.weight', 'multihead_attn_upp.linear_q.weight', 'multihead_attn_low.linear_k.bias', 'multihead_attn_low.linear_v.bias', 'conditioning_layer_phn.bias', 'lm_haed_phn.bias', 'conditioning_layer.weight'] 
