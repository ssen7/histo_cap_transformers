# Debugging Neural Networks: https://benjamin-computer.medium.com/debugging-neural-networks-6fa65742efd
import pytorch_lightning as L
from pytorch_lightning.callbacks import ModelCheckpoint, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
import torchmetrics
from watermark import watermark

import os
import time

import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from dataloader import *
from utils import *
from model_vit_bert import ViTConfigCustom, ViTModelCustom, CustomVEDConfig, CustomVisionEncoderDecoder

# from eval_utils import custom_evaluate, custom_evaluate_only_resnet, custom_evaluate_only_resnet_plus, custom_evaluate_only_vit
from nltk.translate.bleu_score import corpus_bleu

from pytorch_lightning import seed_everything
import argparse
import pdb

from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.trainer_pt_utils import LabelSmoother
from pytorch_lightning.utilities import grad_norm

import warnings

warnings.filterwarnings(
    "ignore", ".*Trying to infer the `batch_size` from an ambiguous collection.*"
)

class LightningModel(L.LightningModule):
    def __init__(self, model, tokenizer, model_lr=1e-3, max_gen_len=128):
        super().__init__()

        self.save_hyperparameters(ignore=["model","tokenizer"])

        self.model_lr = model_lr  # learning rate for encoder if fine-tuning
        self.tokenizer=tokenizer
        self.model = model
        self.max_gen_len=128
        
        self.val_hypotheses=[]
        self.val_references=[]
        
        self.test_hypotheses=[]
        self.test_hypotheses_w_sptokens=[]
        self.test_references=[]
        
        self.val_bleu = torchmetrics.BLEUScore()
        self.test_bleu = torchmetrics.BLEUScore()
    

    def training_step(self, batch, batch_idx):
        pixel_values, labels, attention_mask = batch
        labels, attention_mask = labels.squeeze(0), attention_mask.squeeze(0)
        
        loss = self.model(pixel_values=pixel_values, labels=labels, decoder_attention_mask=attention_mask).loss
        
        self.log("train_loss", loss, on_epoch=True, on_step=False, prog_bar=True)
        
        return loss
    

    def validation_step(self, batch, batch_idx):
        pixel_values, labels, attention_mask = batch
        labels, attention_mask = labels.squeeze(0), attention_mask.squeeze(0)
        
        loss = self.model(pixel_values=pixel_values, labels=labels, decoder_attention_mask=attention_mask).loss
        
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        pixel_values, labels, attention_mask = batch
        labels, attention_mask = labels.squeeze(0), attention_mask.squeeze(0)
        caption_ids = self.model.generate(pixel_values, max_length=self.max_gen_len)
        # print(caption_ids.shape)
        # print(labels.shape)
        caption = self.tokenizer.decode(caption_ids[0], skip_special_tokens=True)
        caption_with_special_tokens = self.tokenizer.decode(caption_ids[0])

        self.test_hypotheses += [caption]
        self.test_hypotheses_w_sptokens += [caption_with_special_tokens]
        self.test_references += [[self.tokenizer.decode(labels[0], skip_special_tokens=True)]]

    def on_test_epoch_end(self):
        references, hypotheses = self.test_references, self.test_hypotheses
        hypwsptokens = self.test_hypotheses_w_sptokens
        # print("Hypotheses: {}".format(hypotheses))
        # print("References: {}".format(references))

        # save preds of test to dataframe
        df=pd.DataFrame()
        df['preds']=hypotheses
        df['target']=[item for sublist in references for item in sublist]
        df['pred_w_sp_tokens']=hypwsptokens
        df.to_csv(os.path.join(self.logger.log_dir, "saved_test.csv"), index=False)
        
        self.test_bleu(hypotheses,references)

        self.log("test_bleu", self.test_bleu, prog_bar=True)
        
        self.test_hypotheses=[]
        self.test_references=[]

    def configure_optimizers(self):
        
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.model_lr)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=8),
            "monitor": "val_loss",
            "frequency": 1,
            "interval":"epoch"
            # If "monitor" references validation metrics, then "frequency" should be set to a
            # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    # def on_before_optimizer_step(self, optimizer):
    # # Compute the 2-norm for each layer
    # # If using mixed precision, the gradients are already unscaled here
    #     norms = grad_norm(self.model, norm_type=2)
    #     self.log_dict(norms)
        
if __name__=='__main__':
    print(watermark(packages="torch,pytorch_lightning,transformers", python=True), flush=True)
    print("Torch CUDA available?", torch.cuda.is_available(), flush=True)

    # df_path='/home/ss4yd/new_lstm_decoder/data_files/prepared_prelim_data_tokenized_cls256_pathcap_thumb_finalv2_scr.pickle'
    # df_path='/home/ss4yd/new_lstm_decoder/data_files/prepared_prelim_data_tokenized_cls256_pathcap_thumb_newsent.pickle'
    df_path='/home/ss4yd/nlp/final_more_female_data.pickle'

    # Hyperparameters
    batch_size=1
    epochs=10
    model_lr=2e-5

    n_layers=5
    
    num_workers=10
    
    # models: Encoder    
    encoder = ViTModelCustom(config=ViTConfigCustom(hidden_size=576), pretrain_4k='vit4k_xs_dino', freeze_4k=True)

    # decoder
    decoder_model_name="emilyalsentzer/Bio_ClinicalBERT"
    decoder = AutoModelForCausalLM.from_pretrained(decoder_model_name, is_decoder=True, add_cross_attention=True)
    tokenizer = AutoTokenizer.from_pretrained(decoder_model_name)

    # encoder decoder model
    model=CustomVisionEncoderDecoder(config=CustomVEDConfig(),encoder=encoder, decoder=decoder, n_layers=n_layers)
    model.config.decoder_start_token_id = tokenizer.cls_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # data loaders
    train_loader = torch.utils.data.DataLoader(ResnetPlusVitDataset(df_path,text_decode_model=decoder_model_name, dtype='train'), 
                                               batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(ResnetPlusVitDataset(df_path,text_decode_model=decoder_model_name, dtype='val'), 
                                             batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(ResnetPlusVitDataset(df_path,text_decode_model=decoder_model_name, dtype='test'), 
                                              batch_size=1, shuffle=False, num_workers=num_workers)

    # lightning configuration
    lightning_model = LightningModel(model, tokenizer, model_lr=model_lr)

    
    callbacks = [
        ModelCheckpoint(save_top_k=1, mode="min", monitor="val_loss", filename='{epoch}-{val_loss:.2f}-{step:.2f}'),  # save top 1 model
        # ModelCheckpoint(save_last=True, filename='{epoch}-{val_bleu:.2f}-{step:.2f}'),  # save last model
        EarlyStopping(monitor="val_loss", min_delta=0.000, patience=4, verbose=False, mode="min"),
        # StochasticWeightAveraging(swa_lrs=1e-2)
    ]

    csv_logger = CSVLogger(save_dir="/scratch/ss4yd/logs_only_vit_bert_fe/", name=f"my_model")
    # ten_logger = TensorBoardLogger(save_dir="/scratch/ss4yd/logs_only_vit_gpt_bert/", name=f"my_model")

    trainer = L.Trainer(
        max_epochs=epochs,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        precision='16-mixed',
        logger=csv_logger,
        log_every_n_steps=100,
        deterministic=False,
        gradient_clip_val=5.0,
        gradient_clip_algorithm="norm",
        accumulate_grad_batches=16,
        # detect_anomaly=True,
        # limit_train_batches=0.05, 
        # limit_val_batches=0.1,
        # limit_test_batches=0.5
    )

    start = time.time()
    trainer.fit(
        model=lightning_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
        # ckpt_path ="/scratch/ss4yd/logs_only_vit_bert/my_model/version_13/checkpoints/epoch=4-val_loss=0.93-step=2880.00.ckpt"
    )

    end = time.time()
    elapsed = end - start
    print(f"Time elapsed {elapsed/60:.2f} min")

    test_bleu = trainer.test(lightning_model, test_loader, ckpt_path="best")

    with open(os.path.join(trainer.logger.log_dir, "outputs.txt"), "w") as f:
        f.write((f"Time elapsed {elapsed/60:.2f} min\n"))
        f.write(f"Test BLEU-4: {test_bleu}")