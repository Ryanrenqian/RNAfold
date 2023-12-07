from datas import RibonanzaDatasetPreTrain # type: ignore
from torch.nn.utils.rnn import pad_sequence
from models import RNAConfig, RNAModel,RNALayerNorm
import torch
import torch.nn as nn
import pandas as pd
from transformers import TrainingArguments, Trainer
from utils import collate_fn
ALL_LAYERNORM_LAYERS = [nn.LayerNorm, RNALayerNorm]


class CustomTrainer(Trainer):
    def get_decay_parameter_names(self, model):
        """
        Get all parameter names that weight decay will be applied to

        Note that some models implement their own layernorm instead of calling nn.LayerNorm, weight decay could still
        apply to those modules since this function only filter out instance of nn.LayerNorm
        """
        decay_parameters = get_parameter_names(model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        return decay_parameters
    
    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = self.get_decay_parameter_names(opt_model)
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    'lr': self.args.learning_rate,
                },
                {
                    "params": [
                        p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    'lr': self.args.learning_rate,
                },
            ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)

            self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

        return self.optimizer
def get_parameter_names(model, forbidden_layer_types):
    """
    Returns the names of the model parameters that are not inside a forbidden layer.
    """
    result = []
    for name, child in model.named_children():
        result += [
            f"{name}.{n}"
            for n in get_parameter_names(child, forbidden_layer_types)
            if not isinstance(child, tuple(forbidden_layer_types))
        ]
    # Add model specific parameters (defined with nn.Parameter) since they are not in any child.
    result += list(model._parameters.keys())
    return result


if __name__ =='__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = RNAConfig()
    df = pd.read_parquet('../datas/data_struct/pretrain_eternerfold.parquet')
    model = RNAModel(config,pretrain=True)
    train_ds = RibonanzaDatasetPreTrain(df, config)
    training_args = TrainingArguments(
        output_dir="./outputs/RoPE_pretrain",
        evaluation_strategy="no",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=int(1),
        learning_rate=1e-3,
        adam_beta2=0.98,
        num_train_epochs=30,
        weight_decay=0.05,
        per_device_train_batch_size=128, 
        load_best_model_at_end=False,
        save_total_limit=10,
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
        eval_dataset=None,
    )
    trainer.train()
    