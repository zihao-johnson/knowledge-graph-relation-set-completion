import math
import random
import itertools
import torch
from torch import nn
def get_collate_fn(
    dic, 
    auto_regression = False
):
    def collate_fn(batch):
        try:
            entity = [example[0] for example in batch]
            batch_input_ids = [torch.tensor(example[1]) for example in batch]
            batch_input_ids = nn.utils.rnn.pad_sequence(batch_input_ids, padding_value=dic["pad"])
            if auto_regression:
                if batch[0][2]:
                    batch_pos_ids = [torch.tensor([dic["sos"]] + example[2] + [dic["eos"]]) for example in batch]
                    batch_pos_ids = nn.utils.rnn.pad_sequence(batch_pos_ids, padding_value=dic["pad"])
                else:
                    batch_pos_ids = [example[2] for example in batch]
            else:
                if batch[0][2]:
                    batch_pos_ids = [torch.tensor(example[2]) for example in batch]
                    batch_pos_ids = nn.utils.rnn.pad_sequence(batch_pos_ids, padding_value=dic["pad"])
                else:
                    batch_pos_ids = [example[2] for example in batch]
            if batch[0][3]:
                batch_neg_ids = [torch.tensor(example[3]) for example in batch]
                batch_neg_ids = nn.utils.rnn.pad_sequence(batch_neg_ids, padding_value=dic["pad"])
            else:
                batch_neg_ids = [example[3] for example in batch]
            batch = {
                "entity": entity,
                "input_ids": batch_input_ids,
                "pos_ids": batch_pos_ids,
                "neg_ids": batch_neg_ids,
            }
        except:
            print(batch[:2])
            assert False    
        return batch
    return collate_fn


def get_data_loader(
    dataset,
    batch_size,
    dic,
    auto_regression=False,
    shuffle=False,
    seed=None
):
    collate_fn = get_collate_fn(dic, auto_regression)

    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=shuffle,
        generator=generator,
    )

    return data_loader