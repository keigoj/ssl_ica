from wav2vec2_module import Wav2vec2Module
from dataloader import MyDataLoader
from torch.utils.data import DataLoader
from dataset_from_manifest import DatasetFromManifest
import argparse
import lightning.pytorch as pl
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
import numpy as np
from datasets import load_metric
from tqdm import tqdm

class DataloaderForPred(pl.LightningDataModule):
    def __init__(self, test_manifest_path, batch_size=8, ngpu=1) -> None:
        super().__init__()
        self.test_manifest_path = test_manifest_path
        self.batch_size = batch_size
        self.ngpu = ngpu
    
    def setup(self, stage: str) -> None:
        self.test_dataset = DatasetFromManifest(self.test_manifest_path)

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.batch_idx_fn)
    
    def batch_idx_fn(self, batch):
        waves, texts = list(zip(*batch))
        max_len = max(len(wave) for wave in waves)
        padded_waves = []
        for wave in waves:
            pad_width = max_len - len(wave)
            padded_wave = np.pad(wave, (0, pad_width), mode="constant", constant_values=0)
            padded_waves.append(padded_wave)
        waves = np.stack(padded_waves)
        return waves, texts


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["base", "att", "att_ica"], required=True)
    parser.add_argument("--name", type=str)
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # set training settings
    train_method = args.method
    model_name = f"{train_method}_{args.name}" if len(args.name) > 0 else f"{train_method}"
    output_model_dir = f"model/{model_name}"
    checkpoint_path = f"{output_model_dir}/{args.checkpoint}"
    checkpoint_path = "/home/hojo/exp/ssl2/model/att_lr1e-4_warm300_30epo/step=12711-wer=0.21800259.ckpt"
    batch_size = 16

    # set the used dataset paths
    test_dataset_path = {"test-clean": "data/test-clean-manifest.json", "test-other": "data/test-other-manifest.json"}

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
    
    trainer = pl.Trainer()
    wer_metric = load_metric("wer", trust_remote_code=True)

    for name, path in test_dataset_path.items():
        data = DataloaderForPred(path, batch_size)
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