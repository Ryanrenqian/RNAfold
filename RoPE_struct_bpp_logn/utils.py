import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    new_batch = dict()
    for k in batch[0].keys():
        if k == 'bbps' or k== 'ids' or k == 'logn':
            continue
        new_batch[k] = pad_sequence((i[k] for i in batch), batch_first=True, padding_value=0) # type: ignore

    length = new_batch['input_ids'].shape[1] # bs ,l, d
    bbps = []
    for i in batch:
        padd_size = length - i['input_ids'].shape[-1]
        bbps.append(F.pad(i['bbps'],(0,padd_size,0,padd_size),value=0))
    new_batch['bbps'] = torch.stack(bbps)
    new_batch['logn'] = torch.Tensor([i['logn'] for i in batch])
    return new_batch