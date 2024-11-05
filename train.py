from pytorch_lightning.loggers import WandbLogger
from wav2vec2_module import Wav2vec2Module
from dataloader import MyDataLoader
from multiprocessing import Value
from typing import Dict
from torch import Tensor
from pytorch_lightning.utilities import rank_zero_info

import torch
import os
import re
import argparse
import lightning.pytorch as pl
import shutil


class HuggingFaceModelCheckpoint(pl.callbacks.ModelCheckpoint):
        def save_huggingface_model(self, trainer, pl_module):
            # Save HuggingFace model
            model = trainer.model if not isinstance(trainer.model, torch.nn.parallel.DistributedDataParallel) else trainer.model.module
            model.model.save_pretrained(f"{self.dirpath}/checkpoint-{trainer.global_step}")

        def _update_best_and_save(
            self, current: Tensor, trainer: "pl.Trainer", monitor_candidates: Dict[str, Tensor]
        ) -> None:
            k = len(self.best_k_models) + 1 if self.save_top_k == -1 else self.save_top_k

            del_filepath = None
            if len(self.best_k_models) == k and k > 0:
                del_filepath = self.kth_best_model_path
                self.best_k_models.pop(del_filepath)

            # do not save nan, replace with +/- inf
            if isinstance(current, Tensor) and torch.isnan(current):
                current = torch.tensor(float("inf" if self.mode == "min" else "-inf"), device=current.device)

            filepath = self._get_metric_interpolated_filepath_name(monitor_candidates, trainer, del_filepath)

            # save the current score
            self.current_score = current
            self.best_k_models[filepath] = current

            if len(self.best_k_models) == k:
                # monitor dict has reached k elements
                _op = max if self.mode == "min" else min
                self.kth_best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
                self.kth_value = self.best_k_models[self.kth_best_model_path]

            _op = min if self.mode == "min" else max
            self.best_model_path = _op(self.best_k_models, key=self.best_k_models.get)  # type: ignore[arg-type]
            self.best_model_score = self.best_k_models[self.best_model_path]

            if self.verbose:
                epoch = monitor_candidates["epoch"]
                step = monitor_candidates["step"]
                rank_zero_info(
                    f"Epoch {epoch:d}, global step {step:d}: {self.monitor!r} reached {current:0.5f}"
                    f" (best {self.best_model_score:0.5f}), saving model to {filepath!r} as top {k}"
                )
            self._save_checkpoint(trainer, filepath)
            self.save_huggingface_model(trainer, trainer.model)

            if del_filepath and self._should_remove_checkpoint(trainer, del_filepath, filepath):
                self._remove_checkpoint(trainer, del_filepath)
                if trainer.is_global_zero: # avoid multiple processes removing the same checkpoint
                    self._del_model(del_filepath)
        
        def _del_model(self, filepath: str):
            """Custom method to delete the model checkpoint along with Hugging Face model"""
            hf_model_dir = os.path.join(self.dirpath, f"checkpoint-{self._get_step_from_filepath(filepath)}")
            if os.path.exists(hf_model_dir):
                # print(f"Deleting Hugging Face model at {hf_model_dir}")
                shutil.rmtree(hf_model_dir)

        def _get_step_from_filepath(self, filepath: str):
            """Helper method to extract the global step from the checkpoint filepath"""
            # example /home/mluser/wav2vec/work_dir/XLS-R/xlsr-step=11-loss=0.00.ckpt
            return int(re.search(r"step=(\d+)-", filepath).group(1))


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", type=str, choices=["base", "att", "att_ica"], required=True)
    parser.add_argument("--name", type=str, default="")
    parser.add_argument("--ngpu", type=int, default=4)
    parser.add_argument("--ica_path", type=str, default=None)
    args = parser.parse_args()
    return args

def main():
    args = get_args()

    # set training settings
    train_method = args.method
    wandb_name = f"{train_method}_{args.name}" if len(args.name) > 0 else f"{train_method}"
    output_model_dir = f"model/{wandb_name}"
    vocab_path = "./data/vocab_subword.json"
    phn_vocab_path = "./data/vocab_phoneme.json"
    lr = 1e-4
    batch_size = 16
    num_epochs = 30
    ngpu = args.ngpu

    # set the used dataset paths
    train_dataset_path = ["data/train-clean-100-manifest.json"]
    valid_dataset_path = ["data/dev-manifest.json"]
    test_dataset_path =  ["data/test-clean-manifest.json", "data/test-other-manifest.json"]
    
    data_module = MyDataLoader(
        train_dataset_path, 
        valid_dataset_path, 
        test_dataset_path, 
        batch_size=batch_size,
        ngpu=ngpu
    )

    num_training_steps = data_module.get_num_training_steps_per_epoch() * num_epochs # 1epoch1gpuあたりのステップ数 * エポック数 190244* epochs
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")
    
    model = Wav2vec2Module(
        model_id=output_model_dir, 
        vocab_path=vocab_path,
        phn_vocab_path=phn_vocab_path,
        train_method=train_method,
        lr=lr, 
        batch_size=batch_size, 
        num_training_steps=num_training_steps, 
        ngpu=ngpu,
        ica_path=args.ica_path,
    )

    wandb_logger = WandbLogger(project="ssl_ica_lt", name=wandb_name)
    model_checkpoint_callback = HuggingFaceModelCheckpoint(
        monitor="val/wer",
        dirpath=output_model_dir,
        filename="step={step}-wer={val/wer:.8f}",
        save_top_k=2,
        mode="min",
        auto_insert_metric_name=False,
        verbose=True,
        save_last=True,
    )

    if ngpu > 1:
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=num_epochs,
            callbacks=[model_checkpoint_callback, lr_monitor],
            logger=wandb_logger,
            val_check_interval=0.5,
            accumulate_grad_batches=1,
            precision=32,
            devices=ngpu,
            strategy="ddp_find_unused_parameters_true",
        )
    else:
        trainer = pl.Trainer(
            accelerator="auto",
            max_epochs=num_epochs,
            callbacks=[model_checkpoint_callback, lr_monitor],
            logger=wandb_logger,
            val_check_interval=0.5,
            accumulate_grad_batches=2,
            precision=32,
            devices=ngpu,
        )
    
    trainer.fit(model, data_module)
    trainer.test(model, data_module)
    
if __name__ == "__main__":
    main()