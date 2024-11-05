from typing import Any, Optional
import lightning.pytorch as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT, OptimizerLRScheduler
from datasets import load_metric
from transformers import get_linear_schedule_with_warmup
from transformers import HubertForCTC, HubertConfig
from hubert_att import HubertForCTCWeighted
from g2p_en import G2p
import torch
import os
import sentencepiece as spm


class HubertModule(pl.LightningModule):
    def __init__(
            self, 
            output_dir,
            vocab_path,
            train_method, 
            num_training_steps, 
            phn_vocab_path=None,
            lr=1e-4, 
            batch_size=8,
            ngpu=4,
            warmup_steps=300,
            ica_path=None,
            processor=None,
            phn_processor=None,
            pretrained_model=None,
        ) -> None:

        super().__init__()
        self.output_dir = output_dir
        self.vocab_path = vocab_path
        self.phn_vocab_path = phn_vocab_path
        self.train_method = train_method
        self.lr = lr
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.validation_outputs = []
        self.ngpu = ngpu
        self.warmup_steps = warmup_steps
        self.ica_path = ica_path
        self.processor = processor
        self.phn_processor = phn_processor
        self.wer_metric = load_metric("wer", trust_remote_code=True)

        # load SSL model
        if self.train_method == "base":
            config = HubertConfig.from_pretrained(
                pretrained_model, 
                ctc_loss_reduction="mean", 
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=self.processor.tokenizer.vocab_size,
            )
            self.model = HubertForCTC.from_pretrained(
                pretrained_model, 
                config=config,
            )
        elif self.train_method == "att" or self.train_method == "att_ica" or self.train_method == "interctc":
            config = HubertConfig.from_pretrained(
                pretrained_model, 
                ctc_loss_reduction="mean", 
                pad_token_id=self.processor.tokenizer.pad_token_id,
                vocab_size=self.processor.tokenizer.vocab_size,
            )
            config.vocab_size = self.processor.tokenizer.vocab_size
            config.phn_vocab_size = self.phn_processor.tokenizer.vocab_size
            config.phn_pad_token_id = self.phn_processor.tokenizer.pad_token_id
            config.phn_vocab_path = self.phn_vocab_path

            self.model = HubertForCTCWeighted.from_pretrained(
                pretrained_model, 
                config=config,
                method=self.train_method,
            )
        self.model.freeze_feature_extractor()

        # load setencepiece model and g2p model
        self.sp = spm.SentencePieceProcessor(model_file="./data/bpe_unigram5000/bpe.model")
        self.g2p = G2p()

        if ica_path is not None:
            self.ica = torch.load(ica_path)
        else:
            self.ica = None

    def forward(self, x, text, phoneme=None) -> Any:
        # Transform input to tensor
        tokenized_text = self.processor.tokenizer(text, return_tensors="pt", padding=True)
        
        # Fill pad_token_id in tokenized_text.input_ids with -100
        tokenized_text.input_ids[tokenized_text.input_ids == self.processor.tokenizer.pad_token_id] = -100

        # To match the type of the model input
        x.input_values = x.input_values.type_as(next(self.model.parameters()))
        x.attention_mask = x.attention_mask.type_as(next(self.model.parameters()))
        tokenized_text.input_ids = tokenized_text.input_ids.type_as(next(self.model.parameters()))

        if self.train_method != "base":
            tokenized_phn = self.phn_processor.tokenizer(phoneme, return_tensors="pt", padding=True)
            tokenized_phn.input_ids[tokenized_phn.input_ids == self.phn_processor.tokenizer.pad_token_id] = -100
            tokenized_phn.input_ids = tokenized_phn.input_ids.type_as(next(self.model.parameters()))
        
        if self.ica is not None:
            ica_mat = {k: v.type_as(next(self.model.parameters())) for k, v in self.ica.items()}
        else:
            ica_mat = None

        # Forward model
        if type(self.model) == HubertForCTC:
            output = self.model(
                x.input_values, 
                attention_mask=x.attention_mask,
                return_dict=True,
                labels=tokenized_text.input_ids, 
                output_hidden_states=False, 
                output_attentions=False,
            )
            return output.logits, output.loss, output.hidden_states, output.attentions
        else:
            output = self.model(
                x.input_values, 
                attention_mask=x.attention_mask,
                return_dict=True,
                labels=tokenized_text.input_ids, 
                phn_labels=tokenized_phn.input_ids,
                output_hidden_states=True, 
                output_attentions=True,
                ica_mat=ica_mat,
            )
            return output["logits"], output["loss"], output["aux_loss"], output["low_loss"], output["upp_loss"]
    
    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=0.005)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.warmup_steps, num_training_steps=self.num_training_steps)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_train_start(self) -> None:
        procesor_dir = os.path.join(self.output_dir, "processor")
        self.processor.tokenizer.save_pretrained(procesor_dir)
        self.processor.save_pretrained(procesor_dir)
        if self.phn_processor is not None:
            phn_processor_dir = os.path.join(self.output_dir, "phn_processor")
            self.phn_processor.tokenizer.save_pretrained(phn_processor_dir)
            self.phn_processor.save_pretrained(phn_processor_dir)

    def get_phoneme(self, texts):
        g2p_list = [self.g2p(text) for text in texts]
        g2p_list = ["|" if phn == " " else phn for phn in g2p_list]
        phoneme = ["".join(phn) for phn in g2p_list]

        return phoneme

    def training_step(self, batch, batch_idx) -> STEP_OUTPUT:
        wave, text = batch
        text = list(text)
        subword = [" ".join(sb) for sb in self.sp.encode(text, out_type=str)]

        if self.train_method == "base":
            logits, loss_ctc, hidden_states, attentions = self.forward(wave, subword)
        elif self.train_method == "att" or self.train_method == "att_ica" or self.train_method == "interctc":
            phoneme = self.get_phoneme(text)
            logits, loss_ctc, aux_loss, low_loss, upp_loss = self.forward(wave, subword, phoneme)
            self.log("train/aux_loss", aux_loss, batch_size=self.batch_size, sync_dist=True)
            
            if low_loss is not None and upp_loss is not None:
                self.log("train/low_loss", low_loss, batch_size=self.batch_size, sync_dist=True)
                self.log("train/upp_loss", upp_loss, batch_size=self.batch_size, sync_dist=True)
        else:
            raise ValueError(f"train_method {self.train_method} is not supported.")
        loss = loss_ctc.mean()

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_str = [pred.replace("▁", " ").lstrip() for pred in self.processor.batch_decode(predicted_ids)]
        wer = self.wer_metric.compute(predictions=predicted_str, references=text)

        self.log("train/wer", wer, batch_size=self.batch_size, sync_dist=True)
        self.log("train/loss", loss, batch_size=self.batch_size, prog_bar=True, sync_dist=True)
        self.log("train/lr", self.trainer.optimizers[0].param_groups[0]["lr"], prog_bar=True, logger=True, sync_dist=True)
        
        return {"loss": loss}

    def validation_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        wave, text = batch
        text = list(text)
        subword = [" ".join(sb) for sb in self.sp.encode(text, out_type=str)]

        if self.train_method == "base":
            logits, loss_ctc, hidden_states, attentions = self.forward(wave, subword)
        elif self.train_method == "att" or self.train_method == "att_ica" or self.train_method == "interctc":
            phoneme = self.get_phoneme(text)
            logits, loss_ctc, aux_loss, low_loss, upp_loss = self.forward(wave, subword, phoneme)
        else:
            raise ValueError(f"train_method {self.train_method} is not supported.")
        loss = loss_ctc.mean()
        
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_str = [pred.replace("▁", " ").lstrip() for pred in self.processor.batch_decode(predicted_ids)]
        wer = self.wer_metric.compute(predictions=predicted_str, references=text)
        
        self.log("val/loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("val/wer", wer, batch_size=self.batch_size, sync_dist=True)
        self.validation_outputs.append({"loss": loss, "wer": torch.tensor(wer)})

        return {"val_loss": loss}
    
    def test_step(self, batch, batch_idx) -> Optional[STEP_OUTPUT]:
        wave, text = batch
        text = list(text)
        subword = [" ".join(sb) for sb in self.sp.encode(text, out_type=str)]

        if self.train_method == "base":
            logits, loss_ctc, hidden_states, attentions = self.forward(wave, subword)
        elif self.train_method == "att" or self.train_method == "att_ica" or self.train_method == "interctc":
            phoneme = self.get_phoneme(text)
            logits, loss_ctc, aux_loss, low_loss, upp_loss = self.forward(wave, subword, phoneme)
        else:
            raise ValueError(f"train_method {self.train_method} is not supported.")
        loss = loss_ctc.mean()
        
        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_str = [pred.replace("▁", " ").lstrip() for pred in self.processor.batch_decode(predicted_ids)]
        wer = self.wer_metric.compute(predictions=predicted_str, references=text)
        
        self.log("test/loss", loss, batch_size=self.batch_size, sync_dist=True)
        self.log("test/wer", wer, batch_size=self.batch_size, sync_dist=True)

        return {"test_loss": loss}
    
    def predict_step(self, batch, dataloader_idx=0) -> Any:
        wave, text = batch
        text = list(text)
        subword = [" ".join(sb) for sb in self.sp.encode(text, out_type=str)]

        if self.train_method == "base":
            logits, loss_ctc, hidden_states, attentions = self.forward(wave, subword)
        elif self.train_method == "att" or self.train_method == "att_ica" or self.train_method == "interctc":
            phoneme = self.get_phoneme(text)
            logits, loss_ctc, aux_loss, low_loss, upp_loss = self.forward(wave, subword, phoneme)
        else:
            raise ValueError(f"train_method {self.train_method} is not supported.")

        predicted_ids = torch.argmax(logits, dim=-1)
        predicted_str = [pred.replace("▁", " ").lstrip() for pred in self.processor.batch_decode(predicted_ids)]

        return predicted_str, text
