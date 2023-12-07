import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import  Trainer
import torch
import torch.nn as nn
from models import RNAConfig, RNAModel,RNALayerNorm
ALL_LAYERNORM_LAYERS = [nn.LayerNorm, RNALayerNorm]

def collate_fn(batch):
    new_batch = dict()
    for k in batch[0].keys():
        if k == 'bbps' or  k == 'logn':
            continue
        if k != 'ids':
            new_batch[k] = pad_sequence((i[k] for i in batch), batch_first=True, padding_value=0) # type: ignore
        else:
            new_batch[k] = pad_sequence((i[k] for i in batch), batch_first=True, padding_value=-1) # type: ignore
    length = new_batch['input_ids'].shape[1] # bs ,l, d
    bbps = []
    for i in batch:
        padd_size = length - i['input_ids'].shape[-1]
        bbps.append(F.pad(i['bbps'],(0,padd_size,0,padd_size),value=0))
    new_batch['bbps'] = torch.stack(bbps)
    new_batch['logn'] = torch.Tensor([i['logn'] for i in batch])
    return new_batch

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
    
