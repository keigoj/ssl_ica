from wav2vec2_module import Wav2vec2Module
<<<<<<< HEAD
from hubert_module import HubertModule
=======
from dataloader import MyDataLoader
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
from torch.utils.data import DataLoader
from dataset_from_manifest import DatasetFromManifest
import argparse
import lightning.pytorch as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import numpy as np
from datasets import load_metric
<<<<<<< HEAD
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2Processor

class DataloaderForPred(pl.LightningDataModule):
    def __init__(self, test_manifest_path, batch_size=8, ngpu=1, processor=None) -> None:
=======
from tqdm import tqdm

class DataloaderForPred(pl.LightningDataModule):
    def __init__(self, test_manifest_path, batch_size=8, ngpu=1) -> None:
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
        super().__init__()
        self.test_manifest_path = test_manifest_path
        self.batch_size = batch_size
        self.ngpu = ngpu
<<<<<<< HEAD
        self.processor = processor
=======
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
    
    def setup(self, stage: str) -> None:
        self.test_dataset = DatasetFromManifest(self.test_manifest_path)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.batch_idx_fn)
    
    def batch_idx_fn(self, batch):
        waves, texts = list(zip(*batch))
<<<<<<< HEAD
        waves = [np.array(wave) for wave in waves]
        padded_waves = self.processor(waves, sampling_rate=16000, return_attention_mask=True, return_tensors="pt", padding=True)

        return padded_waves, texts
=======
        max_len = max(len(wave) for wave in waves)
        padded_waves = []
        for wave in waves:
            pad_width = max_len - len(wave)
            padded_wave = np.pad(wave, (0, pad_width), mode="constant", constant_values=0)
            padded_waves.append(padded_wave)
        waves = np.stack(padded_waves)
        return waves, texts
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["base", "att", "att_ica"], required=True)
<<<<<<< HEAD
    parser.add_argument("--model", type=str, choices=["wav2vec2", "hubert"], required=True)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--checkpoint", type=str, required=False)
=======
    parser.add_argument("--name", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # set training settings
    train_method = args.method
    model_name = f"{train_method}_{args.name}" if len(args.name) > 0 else f"{train_method}"
    output_model_dir = f"model/{model_name}"
<<<<<<< HEAD
    output_model_dir = "/home/hojo/exp/ssl2/model/att_lr0.001_warmup300_epoch30"
    checkpoint_path = f"{output_model_dir}/{args.checkpoint}"
    checkpoint_path = "/home/hojo/exp/ssl2/model/att_lr0.001_warmup300_epoch30/step=222-wer=1.00000000.ckpt"
    batch_size = 16
    vocab_path = "./data/vocab_subword5000.json"
    phn_vocab_path = "./data/vocab_phoneme.json"

    # define transfomers processor
    tokenizer = Wav2Vec2CTCTokenizer(vocab_path, unk_token="<unk>", pad_token="<blank>", word_delimiter_token="▁")
    feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    phn_processor = None

    if train_method != "base":
        phn_tokenizer = Wav2Vec2CTCTokenizer(phn_vocab_path, unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
        phn_processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=phn_tokenizer)
=======
    checkpoint_path = f"{output_model_dir}/{args.checkpoint}"
    checkpoint_path = "/home/hojo/exp/ssl2/model/att_lr1e-4_warm300_30epo/step=12711-wer=0.21800259.ckpt"
    batch_size = 16
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2

    # set the used dataset paths
    test_dataset_path = {"test-clean": "data/test-clean-manifest.json", "test-other": "data/test-other-manifest.json"}

<<<<<<< HEAD
    if args.model == "wav2vec2":
        model = Wav2vec2Module.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=output_model_dir,
            vocab_path=vocab_path,
            phn_vocab_path=phn_vocab_path,
            train_method=train_method,
            num_training_steps=0,
            ngpu=1,
            processor=processor,
            phn_processor=phn_processor,
        )
    elif args.model == "hubert":
        model = HubertModule.load_from_checkpoint(
            checkpoint_path=checkpoint_path,
            output_dir=output_model_dir,
            vocab_path=vocab_path,
            phn_vocab_path=phn_vocab_path,
            train_method=train_method,
            num_training_steps=0,
            ngpu=1,
            processor=processor,
            phn_processor=phn_processor,
        )
    else:
        raise ValueError("Model not supported")
    model.eval()
=======
    model = Wav2vec2Module.load_from_checkpoint(
        checkpoint_path=checkpoint_path,
        model_id=output_model_dir,
        vocab_path="./data/vocab_subword.json",
        phn_vocab_path="./data/vocab_phoneme.json",
        train_method=train_method,
        num_training_steps=0,
        ngpu=1,
    )
    model.eval()
    model.freeze()
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
    
    trainer = pl.Trainer()
    wer_metric = load_metric("wer", trust_remote_code=True)

    for name, path in test_dataset_path.items():
<<<<<<< HEAD
        data = DataloaderForPred(path, batch_size, processor=processor)
=======
        data = DataloaderForPred(path, batch_size)
>>>>>>> 515fa1612f52a5ab0198158e76cfd9e9414f2ec2
        preds = trainer.predict(model, data)
        
        all_hyps, all_refs = [], []
        for hyps, refs in preds:
            all_hyps.extend(hyps)
            all_refs.extend(refs)

        wer_metric = load_metric("wer", trust_remote_code=True)
        wer = wer_metric.compute(predictions=all_hyps, references=all_refs)

        with open(f"{output_model_dir}/result_{name}.txt", "w") as f:
            f.write(f"WER: {wer}\n\n")

            for hyp, ref in zip(all_hyps, all_refs):
                f.write(f"Reference : {ref}\n")
                f.write(f"Hypothesis: {hyp}\n\n")

if __name__ == "__main__":
    main()