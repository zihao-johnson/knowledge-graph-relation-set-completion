
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from copy import deepcopy
import itertools
import os
import time
from datetime import datetime
import math
import random
from scipy.optimize import linear_sum_assignment

from dataloader import get_data_loader


def set_random_seed(seed, deterministic=True):
    import os
    import random
    import numpy as np
    import torch

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Makes PyTorch choose deterministic algorithms where possible.
        # Some operations may raise errors if no deterministic version exists.
        torch.use_deterministic_algorithms(True, warn_only=True)


def read_data(file_dir):
    # load data
    with open(file_dir) as f:
        lines = [e.strip().split("\t") for e in f.readlines()]
    return lines

def load_data(data_loc, data_nam):
    train_lines = read_data(os.path.join(data_loc, data_nam, "train.txt"))
    valid_lines = read_data(os.path.join(data_loc, data_nam, "valid.txt"))
    test_lines = read_data(os.path.join(data_loc, data_nam, "test.txt"))
    rest_line = read_data(os.path.join(data_loc, data_nam, "rest.txt"))
    
    train_size = len(train_lines)
    valid_size = len(valid_lines)
    test_size = len(test_lines)
    print(f"data located at - {data_loc:s}, dataset - {data_nam:s}")
    print("train size: ", train_size)
    print("validation size: ", valid_size)
    print("test size: ", test_size)

    return train_lines, valid_lines, test_lines, rest_line

def save_list_of_lists(file_path, data):
    with open(file_path, "w", encoding="utf-8") as f:
        for row in data:
            # Convert all elements to string and join with tab
            line = "\t".join(map(str, row))
            f.write(line + "\n")

def flatten_list(l):
    # flatten_list from whatever dim to 1 dim
    if type(l) is not list:
        return [l]
    else:
        to_return = []
        for e in l:
            to_return += flatten_list(e)
        return to_return

def get_item2id(data=[], addtional_item=[]):
    # get entity2id map dict
    if data:
        items = sorted(set(flatten_list(data)))
        leng = len(addtional_item)
        to_return = {items[val-leng]:val for val in range(leng,len(items)+leng)}
        for i, k in enumerate(addtional_item):
            to_return[k] = i
    return to_return


def dict_map(example, dic, unknown_token = None):
    # numericalize data...: def numericalize_example(
    if isinstance(example, list):
        return [dict_map(e, dic, unknown_token) for e in example]
    else:
        if unknown_token:
            return dic.get(example, dic.get(unknown_token))
        else:
            return dic.get(example, example)


def count_parameters(model):
    temp =  sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"The model has {temp:,} trainable parameters")


def get_train_samples(
                train_example, 
                No_max_sample, 
                No_half_size=None):
    # simulate train set training dataset:
    example = train_example
    No_relations = len(example) - 1 

    relations = example[1:]
    if No_half_size:
        No_half_size = No_relations - 1
    else:
        No_half_size = math.ceil(No_relations / 2)

    inputs = []
    targets = []
    for i in range(No_max_sample):
        No_sampled_target = random.randint(1,No_relations - No_half_size)
        # No_sampled_target = random.randint(1,1)
        target = set(random.sample(relations,No_sampled_target))
        if target not in targets:
            iput = set(relations).difference(target)
            inputs.append(iput)
            targets.append(target)
    return [[
                example[0],
                list(inputs[i]), 
                list(targets[i])
            ] for i in range(len(targets))]


def save_data(file_dir, data):    
    with open(file_dir, "w") as f:
        f.write("\n".join([" ".join(map(str, e)) for e in data]))


def get_rsc_data_loader(
    args,
    train_lines, 
    validation_lines, 
    test_lines, 
    rest_line,
    relation2id,
    entity2id=None
):
    def save_train_cache(file_path, train_sample, negative_train_sample):
        cache = {
            "seed": args.seed,
            "data_name": args.data_name,
            "No_max_sample": args.No_max_sample,
            "Neg_sample_size": args.Neg_sample_size,
            "num_relations": len(relation2id),
            "ad": args.ad,
            "train_sample": train_sample,
            "negative_train_sample": negative_train_sample,
        }
        torch.save(cache, file_path)

    def load_train_cache(file_path):
        return torch.load(file_path, map_location="cpu")

    # -----------------#
    #   prepare data   #
    # -----------------#
    train_data = dict_map(train_lines, dic=relation2id)
    valid_data = dict_map(validation_lines, dic=relation2id)
    test_data = dict_map(test_lines, dic=relation2id)

    train_dic = {e[0]: e[1:] for e in train_data}

    valid_data = [
        [e[0], train_dic[e[0]], e[1:], []]
        for e in valid_data
        if e[0] in train_dic
    ]

    test_data = [
        [e[0], train_dic[e[0]], e[1:], []]
        for e in test_data
        if e[0] in train_dic
    ]

    # -----------------------------#
    #   load or create train data  #
    # -----------------------------#
    seed = getattr(args, "seed", None)
    if seed is None:
        raise ValueError(
            "args.seed is required for reproducible saved training data. "
            "Please add parser.add_argument('--seed', type=int, default=42)."
        )

    dataset_dir = os.path.join(args.data_loc, args.data_name)
    os.makedirs(dataset_dir, exist_ok=True)

    train_cache_file = os.path.join(
        dataset_dir,
        f"train_cache_seed{seed}_max{args.No_max_sample}_neg{args.Neg_sample_size}.pt"
    )

    # if os.path.exists(train_cache_file):
    if False:
        print(f"Loading saved train cache from: {train_cache_file}")
        cache = load_train_cache(train_cache_file)

        train_sample = cache["train_sample"]
        negative_train_sample = cache["negative_train_sample"]

        if cache.get("No_max_sample") != args.No_max_sample:
            raise ValueError(
                f"Cache mismatch: No_max_sample in cache is {cache.get('No_max_sample')}, "
                f"but args.No_max_sample is {args.No_max_sample}."
            )

        if cache.get("Neg_sample_size") != args.Neg_sample_size:
            raise ValueError(
                f"Cache mismatch: Neg_sample_size in cache is {cache.get('Neg_sample_size')}, "
                f"but args.Neg_sample_size is {args.Neg_sample_size}."
            )

    else:
        print(f"No saved train cache found. Creating: {train_cache_file}")

        train_sample = [
            get_train_samples(e, args.No_max_sample)
            for e in train_data
        ]
        train_sample = list(itertools.chain.from_iterable(train_sample))

        negative_train_sample = [
            get_negative(
                e,
                num_relations=len(relation2id) - len(args.ad),
                K=args.Neg_sample_size,
                device=args.device
            ).tolist()
            for e in train_sample
        ]

        save_train_cache(
            train_cache_file,
            train_sample,
            negative_train_sample
        )

        print(f"Saved train cache to: {train_cache_file}")

    pos_neg_train_sample = [
        train_sample[i] + [negative_train_sample[i]]
        for i in range(len(train_sample))
    ]

    print("train_sample size:", len(train_sample))
    print("pos_neg_train_sample size:", len(pos_neg_train_sample))

    # -----------------#
    #     rest data    #
    # -----------------#
    rest_data = []
    for data in rest_line:
        temp = []
        for e in data[1:]:
            if e in relation2id:
                temp.append(relation2id[e])
        if temp:
            rest_data.append([data[0], temp, [], []])
    print("No of rest_data:", len(rest_data))
    # -----------------#
    #    dataloaders   #
    # -----------------#
    train_data_loader = get_data_loader(
        pos_neg_train_sample,
        args.batch_size,
        relation2id,
        shuffle=True,
        auto_regression=args.auto_regression
    )
    # get train valid and test max length:
    train_data_loader.max_len = max(
        [len(e[1]) for e in pos_neg_train_sample] +
        [len(e[1]) for e in valid_data] +
        [len(e[1]) for e in test_data] +
        [len(e[1]) for e in rest_data]
    )
    
    valid_data_loader = get_data_loader(
        valid_data,
        args.batch_size,
        relation2id,
        auto_regression=args.auto_regression
    )

    if args.decoder_name == "Seq2seq":
        return train_data_loader, valid_data_loader, test_data, rest_data

    test_data_loader = get_data_loader(
        test_data,
        args.batch_size,
        relation2id,
        auto_regression=args.auto_regression
    )
    
    rest_data_loader = get_data_loader(
        rest_data,
        args.batch_size,
        relation2id,
        auto_regression=args.auto_regression
    )

    return train_data_loader, valid_data_loader, test_data_loader, rest_data_loader

    
def info_nce_loss_multi_pos(pos_score, neg_score, tau=0.1):
    """
    pos_score: (P, B)
    neg_score: (K, B)
    """
    P, B = pos_score.shape
    K = neg_score.size(0)

    pos_score = pos_score.unsqueeze(0) 

    # (1, P, B) --> # (1, P, B)
    # (K, 1, B) --> # (K, P, B)
    neg_score = neg_score.unsqueeze(1).expand(K, P, B)

    # 展平成 (1+K, B*P)
    logits = torch.cat(
        [pos_score, neg_score],
        dim= 0
    ).reshape(1 + K, B * P)

    labels = torch.zeros(
        B * P,
        dtype=torch.long,
        device=logits.device
    )
    # print(labels.shape, logits.shape)
    loss = F.cross_entropy(torch.t(logits)/ tau, labels)
    return loss


def prf_at_k(topk_ids, target_sets, dic, ad):
    """
    topk_ids: (B, k)
    target_sets: list[set[int]] length=B
    returns: precision, recall, f1 (scalars)
    """
    B, k = topk_ids.shape
    precisions, recalls, f1s, tp_rrs = [], [], [], []

    for i in range(B):
        pred = set(topk_ids[i].tolist()) - set([dic[e] for e in ad])
        gold = set(target_sets[i].tolist()) - set([dic[e] for e in ad])

        if len(gold) == 0:
            continue

        tp = len(pred & gold)
        precision = tp / k
        recall = tp / len(gold)
        f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

        # MRRs
        reciprocal_sum = 0.0
        tp_count = 0
        
        for rank, rel in enumerate(pred, start=1):
            if rel in gold:
                reciprocal_sum += 1.0 / rank
                tp_count += 1

        if tp_count > 0:
            tp_rr = reciprocal_sum / len(gold)
        else:
            tp_rr = 0.0
        tp_rrs.append(tp_rr)

    return float(sum(precisions)/len(precisions)), float(sum(recalls)/len(recalls)), float(sum(f1s)/len(f1s)), float(sum(tp_rrs) / len(tp_rrs))


def multi_label_convertor(labels):
    labels = labels.tolist()


def train_score(
    model, data_loader, optimizer, criterion, clip, device, pad_idx
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):
        src = batch["input_ids"].to(device)
        pos_trg = batch["pos_ids"].to(device)
        neg_trg = batch["neg_ids"].to(device)
        optimizer.zero_grad()
        pos_score, neg_score = model(src, pos_trg, neg_trg)
        loss = info_nce_loss_multi_pos(pos_score, neg_score, tau=0.1)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def train_MLC(
    model, data_loader, optimizer, criterion, clip, device, pad_idx
):
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):

        src = batch["input_ids"].to(device)
        trg = batch["pos_ids"].to(device)
        optimizer.zero_grad()
        output = model(src)
        trg = trg.transpose(0, 1).contiguous()
        batch_size = trg.size(0)
        num_classes = output.shape[-1]
        multi_hot = torch.zeros(batch_size, num_classes).to(device).scatter_(1, trg, 1.)
        multi_hot[:, pad_idx] = 0.0     
        output_dim = output.shape[-1]

        try:
            loss = criterion(output, multi_hot)
        except:
            print(output, multi_hot)
            loss = criterion(output, multi_hot)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def train_seq(
    model, 
    data_loader, 
    optimizer, 
    criterion, 
    clip, 
    device,
    pad_idx
):
    # pad_index is args in seq model training
    teacher_forcing_ratio = model.args.teacher_forcing_ratio
    model.train()
    epoch_loss = 0
    for i, batch in enumerate(data_loader):

        src = batch["input_ids"].to(device)
        trg = batch["pos_ids"].to(device)
        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio)

        output_dim = output.shape[-1]
        output = output[1:].view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(data_loader)


def make_bipartite_matched_targets_numpy(
    output,
    trg,
    pad_idx,
):
    """
    output: [trg_len, batch_size, vocab_size]
    trg:    [trg_len, batch_size]

    Returns:
        matched_trg: [trg_len - 1, batch_size]
    """

    device = output.device

    # Skip <sos>
    logits = output[1:]      # [T, B, V]
    target = trg[1:]         # [T, B]

    T, B, V = logits.shape

    matched_trg = torch.full(
        (T, B),
        fill_value=pad_idx,
        dtype=torch.long,
        device=device,
    )

    log_probs = torch.log_softmax(logits, dim=-1)  # [T, B, V]

    for b in range(B):
        labels = target[:, b]
        labels = labels[labels != pad_idx]

        # Since this is a set task, remove duplicate labels
        labels = torch.unique(labels)

        m = labels.size(0)

        if m == 0:
            continue

        # scores_torch: [T, m]
        scores_torch = log_probs[:, b, labels]

        # Convert to NumPy
        scores_np = scores_torch.detach().cpu().numpy()

        # Hungarian solves min-cost assignment, so maximize log-prob by minimizing negative log-prob
        cost_np = -scores_np

        row_ind_np, col_ind_np = linear_sum_assignment(cost_np)

        # Convert result back to torch indices
        row_ind = torch.from_numpy(row_ind_np).long().to(device)
        col_ind = torch.from_numpy(col_ind_np).long().to(device)

        matched_trg[row_ind, b] = labels[col_ind]

    return matched_trg


def train_seq_bipartite(
    model, 
    data_loader, 
    optimizer, 
    criterion, 
    clip, 
    device,
    pad_idx
):
    teacher_forcing_ratio = model.args.teacher_forcing_ratio

    model.train()
    epoch_loss = 0

    for i, batch in enumerate(data_loader):
        src = batch["input_ids"].to(device)   # [src_len, B]
        trg = batch["pos_ids"].to(device)     # [trg_len, B]

        optimizer.zero_grad()

        output = model(src, trg, teacher_forcing_ratio)  # [trg_len, B, V]
        output_dim = output.shape[-1]

        matched_trg = make_bipartite_matched_targets_numpy(
            output=output,
            trg=trg,
            pad_idx=pad_idx,
        )  # [trg_len - 1, B]

        output_for_loss = output[1:]  # [trg_len - 1, B, V]

        loss = criterion(
            output_for_loss.reshape(-1, output_dim),
            matched_trg.reshape(-1),
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(data_loader)

    
def evaluate_top_k(
    model,
    data_loader,
    args,
    k=3,
    dic=None,
    save_outputs=False,
    output_prefix="eval",
):
    """
    Evaluate top-k prediction.

    If save_outputs=True, save:
      {output_prefix}_predict.txt
    """

    model.eval()

    all_precisions = []
    all_recalls = []
    all_f1scores = []
    all_tp_rrs = []

    if save_outputs:
        if dic is None:
            raise ValueError("dic must be provided when save_outputs=True.")
        predict = []
        reverse = {value: key for key, value in dic.items()}

    try:
        ad = args.ad
    except:
        ad = []

    with torch.no_grad():
        for batch in data_loader:
            entities = batch["entity"]
            src = batch["input_ids"].to(args.device)     # (L, B)
            if type(batch["pos_ids"][0]) is not type([]):
                has_pos_trg = True
                pos_trg = batch["pos_ids"].to(args.device)  # (L_pos, B)
            else:
                has_pos_trg = False
                pos_trg = None

            topk_ids, topk_scores = model.rank_topk(src, dic["pad"], k)

            if has_pos_trg:
                precision, recall, f1score, tp_rrs = prf_at_k(
                    topk_ids,
                    torch.t(pos_trg),
                    dic,
                    ad,
                )

                all_precisions.append(precision)
                all_recalls.append(recall)
                all_f1scores.append(f1score)
                all_tp_rrs.append(tp_rrs)

            if save_outputs:
                # Convert to batch-first lists.
                pred_b = topk_ids.detach().cpu().tolist()  # already (B, k)
                pred_b = [[entities[i]] + pred_b[i] for i in range(len(entities))]
                predict.extend(pred_b)

    if save_outputs:
        def decode_nested(rows):
            decoded = []
            for row in rows:
                decoded.append([reverse.get(x, str(x)) for x in row])
            return decoded

        predict = decode_nested(predict)
        file_name = os.path.join(args.out_dir, f"{output_prefix}_predict.txt")
        save_list_of_lists(file_name, predict)

    if has_pos_trg:
        if len(all_precisions) == 0:
            return 0.0, 0.0, 0.0, 0.0

        return (
            sum(all_precisions) / len(all_precisions),
            sum(all_recalls) / len(all_recalls),
            sum(all_f1scores) / len(all_f1scores),
            sum(all_tp_rrs) / len(all_tp_rrs),
        )
    else:
        return None



def f1_score(indices, Y):
    pred_set = set(indices)
    true_set = set(Y)
    # Compute intersection
    intersection = pred_set & true_set
    num_common = len(intersection)
    # Precision: common / predicted
    precision = num_common / len(pred_set) if pred_set else 0.0
    # Recall: common / true
    recall = num_common / len(true_set) if true_set else 0.0
    # F1 Score
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    return f1, recall, precision



#  Working on .....................
def get_negative(train_data, num_relations, K, device):
    all_relation = train_data[1] + train_data[2]
    negatives = torch.empty((K), dtype=torch.long, device=device)
    R_e = torch.tensor(list(all_relation), device=device)
    count = 0

    while count < K:
        candidates = torch.randint(
            0, num_relations, (K * 2,), device=device
        )

        mask = ~torch.isin(candidates, R_e)
        valid = candidates[mask]

        num_take = min(K - count, valid.size(0))
        negatives[count:count + num_take] = valid[:num_take]
        count += num_take
    
    return negatives

# def get_negative(train_data, num_relations, K, device, mode):
#     all_relation = train_data[0] + train_data[1]
#     R_e = torch.tensor(list(all_relation), device=device)

#     if mode == "DeepSet":
#         # All possible relations
#         all_relations = torch.arange(num_relations, device=device)

#         # Get relations NOT in R_e
#         negatives = all_relations[~torch.isin(all_relations, R_e)]

#         return negatives

#     # Default behavior (random sampling)
#     negatives = torch.empty((K,), dtype=torch.long, device=device)
#     count = 0

#     while count < K:
#         candidates = torch.randint(0, num_relations, (K * 2,), device=device)
#         mask = ~torch.isin(candidates, R_e)
#         valid = candidates[mask]

#         num_take = min(K - count, valid.size(0))
#         negatives[count:count + num_take] = valid[:num_take]
#         count += num_take

#     return negatives


# def get_negative(train_data, num_relations, K, device):
#     if type(train_data[0]) == type([]) and type(train_data[1]) == type([]):
#         all_relation = train_data[0] + train_data[1]
#     else:
#         all_relation = train_data
#     negatives = torch.empty((K), dtype=torch.long, device=device)
#     R_e = torch.tensor(list(all_relation), device=device)
#     count = 0

#     while count < K:
#         candidates = torch.randint(
#             0, num_relations, (K * 2,), device=device
#         )

#         mask = ~torch.isin(candidates, R_e)
#         valid = candidates[mask]

#         num_take = min(K - count, valid.size(0))
#         negatives[count:count + num_take] = valid[:num_take]
#         count += num_take
    
#     return negatives



# def evaluate_seq(model, 
#                 data_loader, 
#                 dic, 
#                 args,
#                 data_name = "Val",
#                 ): 
    
#     reversed_dic = {value:key for value, key in dic.items()}
#     translations = [model.translate_sentence(
#         sentence[0],
#         model,
#         dic,
#         args.device,
#         max_output_length=8,
#         allow_repeated = False
#     ) for sentence in data_loader]
#     save_list_of_lists("Predicted" + data_name + ".txt", translations)
#     # print(translations[:2])
#     X = [[reversed_dic[e] for e in example[0]] for example in data_loader]
#     predictions = []
#     for translation in translations:
#         temp = []
#         for e in translation:
#           if e == "eos" or e =="sos":
#               pass
#           elif e == "pad":
#             pass
#           else:
#               temp.append(e)
#         predictions.append(temp)
#     # predictions_pure = [list(set(predictions[i]).difference(set(X[i]))) for i in range(len(predictions))]
#     references = [sorted([reversed_dic[e] for e in example[1]]) for example in data_loader]
    
#     print("Preciscions : references")
#     results=[
#     [f1_score(
#         indices=predictions[i], Y=references[i]
#     )] for i in range(len(references))]
#     results = torch.sum(torch.tensor(results), dim = 0)/torch.tensor(results).shape[0]

#     return results.view(-1)


# def translate_sentence(model, data_loader, device, pad_idx, k = 3):
#     model.eval()
#     epoch_loss = 0
#     with torch.no_grad():
#         precisions,recalls,f1scores,tp_rrss = [],[],[],[]
#         for i, batch in enumerate(data_loader):
#             src = batch["en_ids"].to(device)
#             pos_trg = batch["de_ids"].to(device)
#             # neg_trg = batch["neg_ids"].to(device)
#             topk_ids, topk_scores = model.rank_topk(src, pad_idx, k)
#             # print("Evaluated things.......")
#             # print(topk_ids.shape)
#             # print(pos_trg.shape)
#             precision,recall,f1score,tp_rrs = prf_at_k(topk_ids, torch.t(pos_trg), dic, ad)
#             precisions.append(precision)
#             recalls.append(recall)
#             f1scores.append(f1score)
#             tp_rrss.append(tp_rrs)
#             precision = sum(precisions) / len(precisions)
#             recall = sum(recalls) / len(recalls)
#             f1score = sum(f1scores) / len(f1scores)
#             tp_rrs = sum(tp_rrss) / len(tp_rrss)

#     return precision,recall,f1score,tp_rrs



# def close_world_eval(model, data_loader, device, pad_idx, k, dic):
#     model.eval()

#     with torch.no_grad():
#         predict = []
#         known = []
#         entities = []
#         for batch in data_loader:
#             eneity = batch["input_ids"]
#             src = batch["pos_ids"].to(device)
#             topk_ids, topk_scores = model.rank_topk(src, pad_idx, k)

#             src = src.T
#             if topk_ids.shape[1] == src.shape[0]:
#                 topk_ids = topk_ids.T
            
#             topk_ids = topk_ids.clone().detach().cpu().tolist()
#             src = src.cpu().tolist()
#             predict += topk_ids
#             entities += eneity
#             known += src
            
#         reverse = {value: key for key, value in dic.items()}
#         predict = [[reverse[_] for _ in e] for e in predict]
        
#         known = [[reverse[_] for _ in e] for e in known]
#         known = [[entities[i]] + known[i] for i in range(len(entities))]
#         save_list_of_lists("_predict.txt", predict)
#         save_list_of_lists("_known.txt", known)

#  Working on .....................END--------------------------
