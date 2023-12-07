from torch.nn.utils.rnn import pad_sequence
def collate_fn(batch):
    new_batch = dict()
    for k in batch[0].keys():
        if k != 'ids':
            new_batch[k] = pad_sequence((i[k] for i in batch), batch_first=True, padding_value=0) # type: ignore
        else:
            new_batch[k] = pad_sequence((i[k] for i in batch), batch_first=True, padding_value=-1) # type: ignore
    return new_batch