from datas import RibonanzaDatasetTrain # type: ignore
from models import RNAConfig, RNAModel,RNALayerNorm
import torch
import torch.nn as nn
import pandas as pd
from utils import collate_fn, CustomTrainer
from transformers import TrainingArguments
import datetime

if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RNAConfig()
    df = pd.read_parquet('./datas/data_struct/train_data_eternerfold.parquet')
    model = RNAModel(config)
    # set param
    expname = 'RoPE_struct_bpp_logn'
    model_name = 'checkpoint-507600' # 预训练
    now = datetime.datetime.now()
    timestamp = now.strftime("%b%d_%H-%M-%S")
    out = f"./{expname}/outputs/RoPE-{model_name}-{timestamp}"
    print('Pretrained Model:', model_name)
    print('Save Model:', out)
    checkpoint = torch.load(f'./{expname}/pretrain/{model_name}/pytorch_model.bin')
    checkpoint.pop('head.weight')
    checkpoint.pop('head.bias')
    model.load_state_dict(checkpoint,strict=False)
    train_ds = RibonanzaDatasetTrain(df, config,mode='train')
    valid_ds = RibonanzaDatasetTrain(df, config,mode='eval')
    training_args = TrainingArguments(
        output_dir = out,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(1),
        learning_rate=1e-3,
        adam_beta2=0.98,
        num_train_epochs=64,
        weight_decay=0.05,
        per_device_train_batch_size=256, 
        per_device_eval_batch_size=512,
        load_best_model_at_end=True,
        save_total_limit=3,
        report_to=["tensorboard"],
        dataloader_num_workers=16,
        lr_scheduler_type='constant_with_warmup',
        warmup_ratio=0.2
    )
    trainer = CustomTrainer(
        args=training_args, 
        model=model, 
        data_collator=collate_fn, 
        train_dataset=train_ds, 
        eval_dataset=valid_ds
    )
    trainer.train()
    