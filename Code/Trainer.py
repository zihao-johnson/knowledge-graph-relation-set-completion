from Models import *
from utils import *

import torch
import torch.nn as nn
import torch.optim as optim
from SetTransformer import SetTransformer
from Seq2seq import *

class Model_Register(nn.Module):
    def __init__(self, 
                 args, 
                 relation2id):
        super().__init__()
        self.args = args
        self.relation2id = relation2id
        self.encoder_name = args.encoder_name
        self.decoder_name = args.decoder_name
        self.len_relations = len(relation2id)
        self.pad_idx = relation2id["pad"]
        self.encoder_register_list = {
            "RelSetE": RelSetE_Encoder,
            "SetTransformer": SetTransformer,
            "DeepSet": DeepSet,
            "MLP": FlattenMLPEncoder
        }
        self.decoder_register_list = {
            "Score": Scoring_Decoder,
            "MLC": MLC_Decoder,
            "Seq": DeepSet,
            "Seq2seq": "None"
        }
        self.model_handler_register_list = {
            "Score": Score_base,
            "MLC": MLC_base,
            "Seq2seq": create_seq2seq
        }
        if self.decoder_name == "Seq2seq":
            pass
        else:
            self.encoder = self.encoder_register_list[self.encoder_name](
                len_token=self.len_relations,
                pad_idx=self.pad_idx,
                args=args
            )

            self.decoder = self.decoder_register_list[self.decoder_name](
                len_token=self.len_relations,
                pad_idx=self.pad_idx,
                args=args
            )
        
            if self.args.load_embeddings:
                if self.args.encoder_embedding_path is None:
                    raise ValueError(
                        "When args.load_embeddings=True, encoder_embedding_path must be provided."
                    )

                has_decoder_embedding = self.args.decoder_embedding_path is not None

                encoder_exported_embeddings = np.load(self.args.encoder_embedding_path)

                decoder_exported_embeddings = np.load(self.args.decoder_embedding_path) if has_decoder_embedding else None

                import json
                with open(self.args.load_emb_map_dic, "r", encoding="utf-8") as file:
                    exported_dic = json.load(file)

                new_encoder_embedding_tensor = []
                new_decoder_embedding_tensor = [] if has_decoder_embedding else None

                missing_encoder_tokens = []
                missing_decoder_tokens = []

                for k, v in self.relation2id.items():
                    if k in self.args.ad:
                        # -------------------------
                        # Special tokens
                        # -------------------------
                        new_encoder_embedding_tensor.append(
                            self.encoder.embedding.weight.data[v].detach().cpu()
                        )

                        if has_decoder_embedding:
                            new_decoder_embedding_tensor.append(
                                self.decoder.embedding.weight.data[v].detach().cpu()
                            )
                    else:
                        # -------------------------
                        # Normal tokens
                        # -------------------------
                        if k in exported_dic:
                            new_encoder_embedding_tensor.append(
                                torch.tensor(
                                    encoder_exported_embeddings[exported_dic[k]],
                                    dtype=self.encoder.embedding.weight.dtype,
                                )
                            )
                        else:
                            missing_encoder_tokens.append(k)
                            new_encoder_embedding_tensor.append(
                                self.encoder.embedding.weight.data[v].detach().cpu()
                            )

                        if has_decoder_embedding:
                            if k in exported_dic:
                                new_decoder_embedding_tensor.append(
                                    torch.tensor(
                                        decoder_exported_embeddings[exported_dic[k]],
                                        dtype=self.decoder.embedding.weight.dtype,
                                    )
                                )
                            else:
                                missing_decoder_tokens.append(k)
                                new_decoder_embedding_tensor.append(
                                    self.decoder.embedding.weight.data[v].detach().cpu()
                                )

                # Load encoder embedding
                new_encoder_embedding_tensor = torch.stack(new_encoder_embedding_tensor)

                if self.encoder.embedding.weight.data.shape != new_encoder_embedding_tensor.shape:
                    raise ValueError(
                        f"Encoder embedding shape mismatch: "
                        f"model={tuple(self.encoder.embedding.weight.data.shape)}, "
                        f"loaded={tuple(new_encoder_embedding_tensor.shape)}"
                    )

                with torch.no_grad():
                    self.encoder.embedding.weight.copy_(
                        new_encoder_embedding_tensor.to(self.encoder.embedding.weight.device)
                    )

                # -------------------------
                # Case 1: two files, independent embeddings
                # -------------------------
                if has_decoder_embedding:
                    new_decoder_embedding_tensor = torch.stack(new_decoder_embedding_tensor)

                    if self.decoder.embedding.weight.data.shape != new_decoder_embedding_tensor.shape:
                        raise ValueError(
                            f"Decoder embedding shape mismatch: "
                            f"model={tuple(self.decoder.embedding.weight.data.shape)}, "
                            f"loaded={tuple(new_decoder_embedding_tensor.shape)}"
                        )

                    with torch.no_grad():
                        self.decoder.embedding.weight.copy_(
                            new_decoder_embedding_tensor.to(self.decoder.embedding.weight.device)
                        )

                    print("[Embedding] Loaded encoder and decoder embeddings from two files.")
                    print("[Embedding] Encoder and decoder embeddings are independent.")

                # -------------------------
                # Case 2: one file, tied embeddings
                # -------------------------
                else:
                    self.decoder.embedding.weight = self.encoder.embedding.weight

                    print("[Embedding] Loaded one embedding file.")
                    print("[Embedding] Decoder embedding is tied to encoder embedding.")
                    print("[Embedding] Encoder and decoder share updates during training.")

                # -------------------------
                # Zero pad embedding after loading / tying
                # -------------------------
                with torch.no_grad():
                    self.encoder.embedding.weight[self.pad_idx].fill_(0.0)

                    if self.decoder.embedding.weight is not self.encoder.embedding.weight:
                        self.decoder.embedding.weight[self.pad_idx].fill_(0.0)

                # -------------------------
                # Warnings
                # -------------------------
                if missing_encoder_tokens:
                    print(
                        f"[Warning] {len(missing_encoder_tokens)} tokens missing from encoder embedding file. "
                        f"Examples: {missing_encoder_tokens[:10]}"
                    )

                if missing_decoder_tokens:
                    print(
                        f"[Warning] {len(missing_decoder_tokens)} tokens missing from decoder embedding file. "
                        f"Examples: {missing_decoder_tokens[:10]}"
                    )

                # -------------------------
                # Sanity check
                # -------------------------
                if has_decoder_embedding:
                    assert self.decoder.embedding.weight is not self.encoder.embedding.weight
                else:
                    assert self.decoder.embedding.weight is self.encoder.embedding.weight

                self.model = self.model_handler_register_list[self.decoder_name](self.encoder, self.decoder, self.args.device).to(self.args.device)
            else:
                def init_weights(m):
                    for name, param in m.named_parameters():
                        nn.init.uniform_(param.data, -0.1, 0.1)
                self.model = self.model_handler_register_list[self.decoder_name](self.encoder, self.decoder, self.args.device).to(self.args.device)
                self.model.apply(init_weights)
                if args.using_same_embedding:
                    self.model.decoder.embedding.weight = self.encoder.embedding.weight
            print(self.encoder)
            print(self.decoder)
        if self.decoder_name == "Seq2seq":
            self.model = create_seq2seq(
                        self.args, 
                        self.len_relations, 
                        self.relation2id)

        self.optimizer = optim.Adam(self.model.parameters(), lr = self.args.lr)

    def train_process(self, 
                    train_data_loader, 
                    valid_data_loader, 
                    test_data_loader,
                    rest_data_loader):
        if self.decoder_name == "Score":
            criterion = info_nce_loss_multi_pos
            train_fn = train_score
            evaluate_fn = evaluate_top_k
        elif self.decoder_name == "Seq2seq":
            criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
            if self.args.seq_mode == "OTSeq2set":
                train_fn = train_seq_bipartite
            elif self.args.seq_mode == "RL":
                from RLSeq2Seq import train_seq_rl as train_fn
                self.pad_idx = (self.relation2id["sos"], self.relation2id["eos"], self.pad_idx)
            else:    
                train_fn = train_seq
        elif self.decoder_name == "MLC":
            criterion = nn.BCEWithLogitsLoss()
            train_fn = train_MLC
            evaluate_fn = evaluate_top_k
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

        
        # Train Loop:
        # ------------------------------------------------------------
        k__ = [1, 2, 3, 4, 5, 6, 7]
        if self.args.decoder_name == "Score" or self.args.decoder_name == "MLC":
            best_f1score = -float("inf")
            model_save_file = os.path.join(self.args.out_dir, "best_val_f1_model.pt") 
        else:
            model_save_file = self.args.out_dir
            model_name = f"best_valid_{self.args.seq_mode:s}_{self.args.seq_model:s}.pt"
            best_valid_loss = float("inf")
        for epoch in range(self.args.n_epochs):
            train_loss = train_fn(
                model=self.model,
                data_loader=train_data_loader,
                optimizer=self.optimizer,
                criterion=criterion,
                clip=self.args.clip,
                device=self.args.device,
                pad_idx=self.pad_idx
            )

            if self.decoder_name == "Seq2seq":
                valid_loss = self.model.evaluate_fn(
                    data_loader=valid_data_loader,
                    criterion=criterion,
                    data_name="Val",
                    args=self.args
                )
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.model.state_dict(), os.path.join(model_save_file, model_name))
                if epoch % math.ceil(self.args.n_epochs / 10) == 0:
                    print(f"Epoch: {epoch:2d}: Train Loss:{train_loss:2.3f} | Valid Loss: {valid_loss:2.3f}")
            else:
                valid_results = {}
                valid_f1_scores = []

                for k_ in k__:
                    p, r, f, t = evaluate_fn(
                        model=self.model,
                        data_loader=valid_data_loader,
                        k=k_,
                        dic=self.relation2id,
                        save_outputs=False,
                        output_prefix="eval",
                        args=self.args
                    )

                    valid_f1_scores.append(f)
                    valid_results[k_] = [p, r, f, t]

                avg_valid_f1 = sum(valid_f1_scores) / len(valid_f1_scores)

                if best_f1score < avg_valid_f1:
                    best_f1score = avg_valid_f1

                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "best_f1score": best_f1score,
                            "k_list": k__,
                            "selection_metric": "avg_valid_f1_across_k",
                            "valid_results": valid_results,
                            "encoder_name": self.encoder_name,
                            "decoder_name": self.decoder_name,
                            "args": vars(self.args),
                        },
                        model_save_file,
                    )

                if epoch % math.ceil(self.args.n_epochs / 10) == 0:
                    print(f"Epoch: {epoch:2d}: Train Loss:{train_loss:2.3f}")

                    print(
                        f"{self.args.print_pad:s}{self.args.print_pad:s}"
                        "k   On validation set                         |  On test set"
                    )
                    print(
                        f"{self.args.print_pad:s}{self.args.print_pad:s}"
                        "    precision  recall     f1score    tp_rrs   |  "
                        "precision  recall     f1score    tp_rrs"
                    )

                    for k_ in k__:
                        p, r, f, t = valid_results[k_]

                        p_test, r_test, f_test, t_test = evaluate_fn(
                            model=self.model,
                            data_loader=test_data_loader,
                            k=k_,
                            dic=self.relation2id,
                            save_outputs=False,
                            output_prefix="eval",
                            args=self.args
                        )

                        print(
                            f"{self.args.print_pad:s}{self.args.print_pad:s}"
                            f"{k_:<2d}  "
                            f"{p:2.5f}    {r:2.5f}    {f:2.5f}    {t:2.5f}  |  "
                            f"{p_test:2.5f}    {r_test:2.5f}    {f_test:2.5f}    {t_test:2.5f}"
                        )
        # ------------------------------------------------------------
        if self.decoder_name == "Seq2seq":
            print(f"\nEvaludated on valid best model:")
            self.model.load_state_dict(torch.load(os.path.join(model_save_file, model_name)))
            [F1, recall, precision] = self.model.evaluation(
                                                data = test_data_loader, 
                                                sos_token=self.relation2id["sos"], 
                                                eos_token=self.relation2id["eos"], 
                                                dic=self.relation2id,
                                                device=self.args.device,
                                                save_outputs=self.args.save_outputs).tolist()
            print(f"Test  set results - precision : {precision}, recall : {recall}, F1 score : {F1}")
        else:
            # eval after training:
            k__ = [1,2,3,4,5,6]
            print(f"{self.args.print_pad:s}{self.args.print_pad:s}{self.args.print_pad:s}{self.args.print_pad:s}Best model on test set")
            print (f"{self.args.print_pad:s}{self.args.print_pad:s}precision  recall     f1score    tp_rrs")
            state_dict = torch.load(model_save_file, weights_only=False)
            self.model.load_state_dict(state_dict["model_state_dict"])
            self.model.eval()
            for k_ in k__:
                if_save_outputs = True if k_ == k__[-1] else False
                test_p,test_r,test_f,test_t = evaluate_fn(
                    model=self.model,
                    data_loader=test_data_loader,
                    k=k_,
                    dic=self.relation2id,
                    save_outputs=if_save_outputs,
                    output_prefix="rest",
                    args=self.args
                    )
                print(f"{self.args.print_pad:s}{self.args.print_pad:s}{test_p:2.5f}    {test_r:2.5f}    {test_f:2.5f}    {test_t:2.5f}")
            
            print("Rest saving to here:")
            evaluate_fn(
                        model=self.model,
                        data_loader=rest_data_loader,
                        k=k__[-1],
                        dic=self.relation2id,
                        save_outputs=True,
                        output_prefix="rest",
                        args=self.args)

            print("\n------------END program------------") 