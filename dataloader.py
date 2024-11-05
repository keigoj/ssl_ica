from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from dataset_from_manifest import DatasetFromManifest
import lightning.pytorch as pl
import numpy as np
import math

class MyDataLoader(pl.LightningDataModule):
    def __init__(
            self,
            train_manifest_path,
            valid_manifest_path,
            test_manifest_path,
            batch_size=8,
            ngpu=1,
            processor=None
        ) -> None:
        super().__init__()
        self.train_manifest_path = train_manifest_path
        self.valid_manifest_path = valid_manifest_path
        self.test_manifest_path = test_manifest_path
        self.batch_size = batch_size
        self.ngpu = ngpu
        self.processor = processor
        self.train_dataset = DatasetFromManifest(self.train_manifest_path)
    
    def setup(self, stage: str) -> None:
        if stage == "fit" or stage is None:
            self.valid_dataset = DatasetFromManifest(self.valid_manifest_path)
        elif stage == "test" or stage is None:
            self.test_dataset = DatasetFromManifest(self.test_manifest_path)
        elif stage == "predict" or stage is None:
            if isinstance(self.test_manifest_path, list):
                self.test_dataset = [DatasetFromManifest(path) for path in self.test_manifest_path]
            else:
                self.test_dataset = DatasetFromManifest(self.test_manifest_path)
    
    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4, collate_fn=self.batch_idx_fn)
    
    def val_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.valid_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.batch_idx_fn)
    
    def test_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.batch_idx_fn)
    
    def predict_dataloader(self) -> EVAL_DATALOADERS:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=4, collate_fn=self.batch_idx_fn)

    def batch_idx_fn(self, batch):
        waves, texts = list(zip(*batch))
        waves = [np.array(wave) for wave in waves]
        padded_waves = self.processor(waves, sampling_rate=16000, return_attention_mask=True, return_tensors="pt", padding=True)

        return padded_waves, texts

    def get_num_training_steps_per_epoch(self):
        return math.ceil(len(self.train_dataset) / (self.batch_size * self.ngpu))

