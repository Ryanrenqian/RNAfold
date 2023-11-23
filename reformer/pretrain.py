from data import preprocess,SPL_Dataset
from transformers import DataCollatorForLanguageModeling
from transformers import RoFormerForMaskedLM,RoFormerConfig
from transformers import Trainer, TrainingArguments
from transformers import PreTrainedTokenizerFast
from transformers import LineByLineTextDataset
import torch,os
import numpy as np
import pandas as pd
if __name__ =='__main__':
    # os.environ['CUDA_VISIBLE_DEVICES']='5,7'
    # train_data_path = '../datas/pretrain_data.npy'

    train_data_path = '../datas/pretrain_seqs.txt'
    tokenizer = PreTrainedTokenizerFast(tokenizer_file="./my_tokens")
    tokenizer.mask_token = "[MASK]"
    tokenizer.pad_token = "[PAD]"
    print(tokenizer)
    if not os.path.exists(train_data_path):
        df1 = pd.read_parquet('../datas/train_data.parquet')
        df2 = pd.read_parquet('../datas/test_sequences.parquet')
        df = pd.concat([df1[['sequence_id','sequence']][:len(df1)//2],df2[['sequence_id','sequence']]])
        with open(train_data_path,'w') as f:
            f.writelines([seq+'\n' for seq in df['sequence']])

    dataset = LineByLineTextDataset(
        tokenizer = tokenizer,
        file_path='../datas/pretrain_seqs.txt',
        block_size = 457
        # cache_dir = '../datas/cache/pretrain/'
        )
    

    config = RoFormerConfig(
        vocab_size=28,
        max_position_embeddings=457,
        num_hidden_layers=12,    #L
        hidden_size=192,        #H
        intermediate_size= 768,
        num_attention_heads=6,  #A
        type_vocab_size=2,
    )
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    model = RoFormerForMaskedLM(config=config)
    training_args = TrainingArguments(
        output_dir='./outputs/pretrain_model/',
        overwrite_output_dir=True,
        num_train_epochs=200,
        per_device_train_batch_size=64,
        save_steps=10000,
        save_total_limit=10,
        prediction_loss_only=True,
        max_steps=0,
        learning_rate=1e-4,
        adam_beta1=0.9,
        adam_beta2=0.99,
        weight_decay=0.05,
        lr_scheduler_type='linear',
        warmup_steps=10000,
        report_to="tensorboard",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
    )
    # Train
    print("Pre-training BERT model")
    trainer.train()


    # Save model
    model_path = './outputs/pretrain_mlm/final_pretrain.pt'
    print("Saving model at", model_path)
    trainer.save_model(model_path)
    
    
        
    
    
