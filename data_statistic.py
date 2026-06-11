import os
from utils import *
import json
import argparse
from collections import Counter

def statistic(args, data, rel_fre_path):
    with open(rel_fre_path, "r", encoding="utf-8") as f:
            relation_frequency = json.load(f)

    return_ = [rel_fre_path]
    for e in data:
        return_.append(f"{args.data_split[i]} Entity_No: {str(len(data[i])):s}")
        No_R_e = [len(e_)-1 for e_ in e]
        return_.append(f"{args.data_split[i]} |R_e|: Max-{str(max(No_R_e)):s}, Min-{str(min(No_R_e)):s}, Mean-{str(sum(No_R_e)/len(No_R_e)):s}")
        all_rel = flatten_list([e_[1:] for e_ in e])
        R_e_freq = Counter(all_rel)
        mean_freq = sum(R_e_freq.values()) / len(R_e_freq.values())
        return_.append(
            f"{args.data_split[i]} R_e_pair: Max-{str(max(R_e_freq.values())):s}, Min-{str(min(R_e_freq.values())):s}, Mean-{str(mean_freq):s}"
            )

        R_freq = {key: value for key, value in relation_frequency.items() if key in all_rel}
        Min = min(R_freq.values())
        Max = max(R_freq.values())
        Mean = sum(R_freq.values())/len(R_freq.values())
        return_.append(
            f"{args.data_split[i]} R_freq: Max-{str(Max):s}, Min-{str(Min):s}, Mean-{str(Mean):s}"
            )
        return return_
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dataset_dir", default="../Data")
    parser.add_argument("--dataset_name", default=["FB15k-237", "Nell-995", "Nell-1115"])
    parser.add_argument("--data_split", default=["train", "valid", "test", "rest"])
    args = parser.parse_args()

    
    data_loc = [os.path.join(args.dataset_dir, data_name) for data_name in args.dataset_name]
    rel_fre_path = [os.path.join(p, "relation_frequency.json") for p in data_loc]        
    rel_data = [load_data(args.dataset_dir, data_name) for data_name in args.dataset_name] 

    for i in range(len(args.dataset_name)):
        for e in statistic(args, rel_data[i], rel_fre_path[i]):
            print(e)
