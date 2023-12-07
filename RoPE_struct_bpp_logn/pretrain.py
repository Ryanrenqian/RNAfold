from datas import RibonanzaDatasetPreTrain # type: ignore
from torch.nn.utils.rnn import pad_sequence
from models import RNAConfig, RNAModel,RNALayerNorm
import torch
import torch.nn as nn
import pandas as pd

import datetime
from utils import collate_fn, CustomTrainer
from transformers import TrainingArguments



if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RNAConfig()
    df = pd.read_parquet('./datas/data_struct/pretrain_eternerfold.parquet')
    model = RNAModel(config,pretrain=True)
    expname = 'RoPE_struct_bpp_logn'
    now = datetime.datetime.now()
    timestamp = now.strftime("%b%d_%H-%M-%S")
    out = f"./{expname}/pretrain/{timestamp}"
    train_ds = RibonanzaDatasetPreTrain(df, config)
    training_args = TrainingArguments(
        output_dir=out,
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(1),
        learning_rate=1e-3,
        adam_beta2=0.98,
        num_train_epochs=5,
        weight_decay=0.05,
        per_device_train_batch_size=128, 
        load_best_model_at_end=False,
        save_total_limit=10,
        report_to=["tensorboard"],
        dataloader_num_workers=16,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=0.1
    )
    trainer = CustomTrainer(
        args=training_args, 
        model=model, 
        data_collator=collate_fn, 
        train_dataset=train_ds, 
        eval_dataset=None,
    )
    trainer.train()
    