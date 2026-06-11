from datetime import datetime
import time
import random
import argparse
import os

import torch

import numpy as np

from utils import *
from Models import RelSetE_Encoder, Scoring_Decoder, Score_base 
from Trainer import *



if __name__ == "__main__":

    # test_mode = True
    # test_mode = False

    parser = argparse.ArgumentParser(description="")

    # env setting:
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--print_pad", default="  ")

    # datases related:
    # parser.add_argument()
    parser.add_argument("--data_loc", default="../Data/", help="dataset location")
    parser.add_argument("--data_name", default="FB15k-237", help="KG dataset name")

    parser.add_argument("--original_data", default="all.txt", help="original kg data ")
    parser.add_argument("--train_file", default="train.txt", help="Clearned train dataset")
    parser.add_argument("--valid_file", default="valid.txt", help="Clearned valid dataset")
    parser.add_argument("--test_file", default="test.txt", help="Clearned test dataset")
    parser.add_argument("--resl_file", default="rest.txt", help="Clearned rest dataset")
    
    parser.add_argument("--No_max_sample", default=4, type = int, help="self-supervised training samples No per entry")
    parser.add_argument("--output_location", default="../Logs", help="Input file name")
    parser.add_argument("--use_entity", default=False, help="Specify if use entities as input")

    # pretrained model related
    parser.add_argument("--load_embeddings", default=False, help="")
    parser.add_argument("--load_emb_map_dic", default=None, help="loaded embedding maps information ")
    parser.add_argument("--encoder_embedding_path", default=None, help="Location of pre trained kge models")
    parser.add_argument("--decoder_embedding_path", default=None, help="Location of pre trained kge models")

    # parser.add_argument("--random_split", default=False, help="")
    # parser.add_argument("--nonexist_filter", default=False, help="")
    parser.add_argument("--using_same_embedding", default=True, help="True if encoder and decoder shares relations embeddings")
    
    # models related
    parser.add_argument("--encoder_name", default="", help="")
    parser.add_argument("--decoder_name", default="", help="")
    # parser.add_argument("--model_config", default=dict(), help="models and its hyperparameters specs")
    parser.add_argument("--Neg_sample_size", type = int, default = 31)
    parser.add_argument("--save_negatives", default=False)
    parser.add_argument("--save_outputs", default=False)

    # training related:
    parser.add_argument("--batch_size", type = int, default = 128)
    parser.add_argument("--n_epochs", type = int, default = 70)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--clip", type=int, default=1.0)

    parser.add_argument("--auto_regression", default=True, help="")

    # outcome saving realted:
    parser.add_argument("--out_loc", default="../Logs/", help="")
    parser.add_argument("--best_model_dir", default="best_model", help="")
    parser.add_argument("--seed", type=int, default=10)

    parser.add_argument("--seq_mode", default="rpw", help="seqence mode [orig|rpw]")
    parser.add_argument("--seq_model", default="GRU", help="seqence model [RNN|LSTM|GRU]")
    parser.add_argument("--teacher_forcing_ratio", default=0.2, help="")

    parser.add_argument(
        "--sequential",
        action="store_true",
        help="Use sequential/causal masking in the encoder or decoder.",
    )
    parser.add_argument(
        "--positional",
        action="store_true",
        help="Use positional encoding.",
    )
    # program test
    parser.add_argument("--test_No", default=None, help="")


    args = parser.parse_args()
    args.seed = random.randint(0,10000)
    args.device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    args.ad = ["sos","eos","pad"] if args.decoder_name == "Seq2seq" else ["pad"]
    args.auto_regression = True if args.decoder_name == "Seq2seq" else False
    if args.encoder_name == "Seq2seq" or args.decoder_name == "Seq2seq":
        if args.seq_mode and args.seq_model:
            print(f"Using {args.seq_mode:s} {args.seq_model:s}...")
        else:
            assert False, f"{args.seq_mode:s} or {args.seq_model:s} is not specified !"
    else:
        args.seq_mode = False 
        args.seq_model = False 

    if args.decoder_name == "MLC":
        args.using_same_embedding = False 
    
    if args.seq_mode == "OTSeq2set":
        args.seq_model = "GRU" 
    
    out_parts = [
        args.out_loc,
        args.data_name,
        f"{args.encoder_name}-{args.decoder_name}",
        f"seed_{args.seed}",
    ]
    if args.decoder_name == "Score":
        out_parts.append(f"neg_{args.Neg_sample_size}")
    args.out_dir = os.path.join(*out_parts)
    
    print(args)
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Experiments on Encoder: {args.encoder_name} - Decoder: {args.decoder_name} on args.device")
    print(f"Outcome saving to folder: {args.out_dir}")
    set_random_seed(args.seed)


    train_lines, validation_lines, test_lines, rest_line = load_data(args.data_loc, args.data_name)
    
    all_entities = sorted(list(set([e[0] for e in train_lines])))
    all_relations = sorted(
        list(
            set(flatten_list([e[1:] for e in train_lines] + [e[1:] for e in validation_lines] + [e[1:] for e in test_lines]))
            )
        )
    relation2id = get_item2id(list(all_relations), args.ad)
    entity2id = get_item2id(list(all_entities), [])
    for key in args.ad:
        print(f'Index of {key:s} = {relation2id[key]:d}')

    train_data_loader, valid_data_loader, test_data_loader, rest_data_loader = get_rsc_data_loader(
                        args,
                        train_lines, 
                        validation_lines, 
                        test_lines, 
                        rest_line,
                        relation2id
                        )
    
    if args.encoder_name == "MLP":
        args.max_len = train_data_loader.max_len

    trainer = Model_Register(args, relation2id)
    trainer.train_process(
        train_data_loader,
        valid_data_loader,
        test_data_loader,
        rest_data_loader
    )
    



