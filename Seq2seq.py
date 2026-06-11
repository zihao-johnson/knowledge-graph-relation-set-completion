import random
import torch
import torch.nn as nn
import numpy as np
from utils import f1_score

class Attention(nn.Module):
    def __init__(self, encoder_hidden_dim, decoder_hidden_dim):
        super().__init__()
        self.attn_fc = nn.Linear(
            (encoder_hidden_dim * 2) + decoder_hidden_dim, decoder_hidden_dim
        )
        self.v_fc = nn.Linear(decoder_hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        # hidden = [batch size, decoder hidden dim]
        # encoder_outputs = [src length, batch size, encoder hidden dim * 2]
        batch_size = encoder_outputs.shape[1]
        src_length = encoder_outputs.shape[0]
        # repeat decoder hidden state src_length times
        hidden = hidden.unsqueeze(1).repeat(1, src_length, 1)
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # hidden = [batch size, src length, decoder hidden dim]
        # encoder_outputs = [batch size, src length, encoder hidden dim * 2]
        energy = torch.tanh(self.attn_fc(torch.cat((hidden, encoder_outputs), dim=2)))
        # energy = [batch size, src length, decoder hidden dim]
        attention = self.v_fc(energy).squeeze(2)
        # attention = [batch size, src length]
        return torch.softmax(attention, dim=1)

class Encoder(nn.Module):
    def __init__(self, No_tokens, embedding_dim, hidden_dim, n_layers, dropout, rnn_type="GRU"):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(No_tokens, embedding_dim)
        if rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        elif rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        elif rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.rnn_type = rnn_type

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        embedded = self.dropout(self.embedding(src))
        # embedded = [src length, batch size, embedding dim]
        if self.rnn_type == "LSTM":
            outputs, (hidden, cell) = self.rnn(embedded)
            hidden = (hidden, cell)
        elif self.rnn_type == "RNN":
            outputs, hidden = self.rnn(embedded)
        elif self.rnn_type == "GRU":
            outputs, hidden = self.rnn(embedded)
        return hidden

class ReadProcessWriteEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        embedding_dim,
        hidden_dim,
        n_layers,
        dropout,
        rnn_type="GRU",
        processing_steps=3,
        pad_idx=0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.rnn_type = rnn_type
        self.processing_steps = processing_steps
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=pad_idx)
        self.memory_proj = nn.Linear(embedding_dim, hidden_dim)

        process_input_dim = 2 * hidden_dim

        if rnn_type == "LSTM":
            self.process_cell = nn.LSTMCell(process_input_dim, hidden_dim)
        elif rnn_type == "GRU":
            self.process_cell = nn.GRUCell(process_input_dim, hidden_dim)
        elif rnn_type == "RNN":
            self.process_cell = nn.RNNCell(process_input_dim, hidden_dim)
        else:
            raise ValueError(f"Unsupported rnn_type: {rnn_type}")

        self.hidden_fc = nn.Linear(2 * hidden_dim, hidden_dim)
        if self.rnn_type == "LSTM":
            self.cell_fc = nn.Linear(2 * hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src length, batch size]
        device = src.device
        src_len, batch_size = src.shape

        # [L, B, E]
        embedded = self.dropout(self.embedding(src))

        # memory: [L, B, H]
        memories = self.memory_proj(embedded)

        # Do not let padded positions participate in read attention.
        # padding_idx makes the pad embedding zero, but memory_proj has a bias,
        # so projected pad vectors can otherwise become non-zero.
        src_mask = src.ne(self.pad_idx)  # [L, B]
        memories = memories * src_mask.unsqueeze(-1).float()

        h_t = torch.zeros(batch_size, self.hidden_dim, device=device)
        q_star_t = torch.zeros(batch_size, 2 * self.hidden_dim, device=device)

        if self.rnn_type == "LSTM":
            c_t = torch.zeros(batch_size, self.hidden_dim, device=device)

        for _ in range(self.processing_steps):
            if self.rnn_type == "LSTM":
                h_t, c_t = self.process_cell(q_star_t, (h_t, c_t))
            else:
                h_t = self.process_cell(q_star_t, h_t)

            # scores: [L, B]
            scores = torch.einsum("lbh,bh->lb", memories, h_t)
            scores = scores.masked_fill(~src_mask, torch.finfo(scores.dtype).min)
            attn = torch.softmax(scores, dim=0)  # [L, B]
            attn = attn * src_mask.float()
            attn = attn / attn.sum(dim=0, keepdim=True).clamp(min=1e-12)

            # read vector: [B, H]
            r_t = torch.sum(attn.unsqueeze(-1) * memories, dim=0)

            # controller input for next step: [B, 2H]
            q_star_t = torch.cat([h_t, r_t], dim=1)

        # final set representation
        set_repr = q_star_t  # [B, 2H]

        # map to decoder-compatible hidden
        hidden_last = self.hidden_fc(set_repr) # [B, H]
        hidden = hidden_last.unsqueeze(0).repeat(self.n_layers, 1, 1)

        if self.rnn_type == "LSTM":
            cell_last = self.cell_fc(set_repr)  # [B, H]
            cell = cell_last.unsqueeze(0).repeat(self.n_layers, 1, 1)
            return (hidden, cell)

        return hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, embedding_dim, hidden_dim, n_layers, dropout, rnn_type, pad_idx):
        super().__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad_idx)
        # self.embedding = nn.Embedding(output_dim, embedding_dim, padding_idx=pad_idx)
        self.rnn_type = rnn_type
        if self.rnn_type == "LSTM":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        if self.rnn_type == "GRU":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        if self.rnn_type == "RNN":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, n_layers, dropout=dropout)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden):

        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))

        if self.rnn_type == "LSTM":
            (hidden, cell) = hidden
            output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
            hidden = (hidden, cell)
        if self.rnn_type == "GRU" or self.rnn_type == "RNN":
            output, hidden = self.rnn(embedded, hidden)
        prediction = self.fc_out(output.squeeze(0))
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, args):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = args.device
        self.args = args
        assert (
            encoder.hidden_dim == decoder.hidden_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"
        assert (
            encoder.n_layers == decoder.n_layers
        ), "Encoder and decoder must have equal number of layers!"

    def forward(self, src, trg, teacher_forcing_ratio):
        # teacher_forcing_ratio is probability to use teacher forcing
        # e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)
        # first input to the decoder is the <sos> tokens
        
        # input = [batch size]
        input = trg[0, :]
        for t in range(1, trg_length):
            # assuming not sequential shift
            # insert input token embedding, previous hidden and previous cell states
            # receive output tensor (predictions) and new hidden and cell states
            output, hidden = self.decoder(input, hidden)
            # place predictions in a tensor holding predictions for each token
            outputs[t] = output
            # decide if we are going to use teacher forcing or not
            teacher_force = random.random() < teacher_forcing_ratio
            # get the highest predicted token from our predictions
            top1 = output.argmax(1)
            # if teacher forcing, use actual next token as next input
            # if not, use predicted token
            input = trg[t] if teacher_force else top1
        return outputs

    def transform(self, src, trg, teacher_forcing_ratio):
        batch_size = trg.shape[1]
        trg_length = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        # tensor to store decoder outputs
        outputs = torch.zeros(trg_length, batch_size, trg_vocab_size).to(self.device)
        # last hidden state of the encoder is used as the initial hidden state of the decoder
        hidden = self.encoder(src)
        input = trg[0, :]
        # input = [batch size]
        for t in range(1, trg_length):
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)
            input = trg[t] if teacher_force else top1
            # input = [batch size]
        return outputs

    def train_fn(self,
        model, 
        data_loader, 
        optimizer, 
        criterion, 
        clip, 
        device,
        pad_idx
    ):
        teacher_forcing_ratio = self.args.teacher_forcing_ratio
        model.train()
        epoch_loss = 0
        for i, batch in enumerate(data_loader):

            src = batch["input_ids"].to(device)
            trg = batch["pos_ids"].to(device)
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio)

            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg.view(-1)
            loss = criterion(output, trg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / len(data_loader)

    def evaluate_fn(self, data_loader, criterion, data_name, args):
        self.eval()
        epoch_loss = 0.0

        with torch.no_grad():
            for batch in data_loader:
                src = batch["input_ids"].to(args.device)
                trg = batch["pos_ids"].to(args.device)

                # trg: (T, B)
                # decoder input usually uses trg[:-1]
                # gold target usually uses trg[1:]
                output = self.forward(src, trg, teacher_forcing_ratio=0.0)

                # If output shape is (T, B, V), align it with trg.
                output_dim = output.shape[-1]

                # Common seq2seq convention:
                # output predicts tokens from step 1 onward.
                output = output[1:].reshape(-1, output_dim)
                gold = trg[1:].reshape(-1)

                loss = criterion(output, gold)
                epoch_loss += loss.item()

        return epoch_loss / max(1, len(data_loader))

    def translate_sentence(self,
        sentence,
        dic,
        sos_token,
        eos_token,
        device,
        max_output_length=15,
        allow_repeated = True
    ):
        self.eval()
        with torch.no_grad():
            tensor = torch.LongTensor(sentence).unsqueeze(-1).to(device)
            hidden = self.encoder(tensor)
            inputs = [sos_token]
            for _ in range(max_output_length):
                inputs_tensor = torch.LongTensor([inputs[-1]]).to(device)
                output, hidden = self.decoder(inputs_tensor, hidden)
                predicted_token = output.argmax(-1).item()
                if allow_repeated:
                    i = 0
                    while predicted_token in inputs + sentence:
                        output[0][predicted_token] = 0
                        predicted_token = output.argmax(-1).item()
                        i+=1
                        if i == 10:
                            break
                else:
                    i = 0
                    while predicted_token in sentence:
                        output[0][predicted_token] = 0
                        predicted_token = output.argmax(-1).item()
                        i+=1
                        if i == 10:
                            break
                inputs.append(predicted_token)
                if predicted_token == eos_token:
                    break
            reversed_dic = {value:key for key, value in dic.items()}
            tokens = [reversed_dic[i] for i in inputs]
        return tokens

    def evaluation(self, data, sos_token, eos_token, dic, device, save_outputs, data_name = "Val"): 
        translations = [self.translate_sentence(
            sentence[1],
            dic,
            sos_token,
            eos_token,
            device,
            max_output_length=8,
            allow_repeated = False
        ) for sentence in data]
        if save_outputs:
            save_list_of_lists("Predicted" + data_name + ".txt", translations)
        reversed_dic = {key:value for value, key in dic.items()}
        
        predictions = []
        for translation in translations:
            temp = []
            for e in translation:
                if e == "eos" or e =="sos":
                    pass
                elif e == "pad":
                    pass
                else:
                    temp.append(e)
            predictions.append(temp)
        try:
            references = [sorted([reversed_dic[e] for e in example[2]]) for example in data]
        except:
            print(reversed_dic)
            print("\n")
            print(dic)
            assert False, "key error in relation2dic. "
        print("Preciscions : references")
        results=[
        [f1_score(
            indices=predictions[i], Y=references[i]
        )] for i in range(len(references))]
        results = torch.sum(torch.tensor(results), dim = 0)/torch.tensor(results).shape[0]

        return results.view(-1)


def create_seq2seq(args, No_tokens, dic):
    # model related
    defaults = {
        "encoder_embedding_dim": 256,
        "decoder_embedding_dim": 256,
        "hidden_dim": 512,
        "n_layers": 2,
        "encoder_dropout" : 0.4,
        "decoder_dropout" : 0.4,
        }
    rnn_type = args.seq_model
    mode = args.seq_mode
    device = args.device
    # rnn_type = model_cfg["rnn_type"]
    model_cfg = {**defaults, **getattr(args, "model_config", {})}

    encoder_embedding_dim = model_cfg["encoder_embedding_dim"]
    decoder_embedding_dim = model_cfg["decoder_embedding_dim"]
    hidden_dim = model_cfg["hidden_dim"]
    n_layers = model_cfg["n_layers"]
    encoder_dropout = model_cfg["encoder_dropout"]
    decoder_dropout = model_cfg["decoder_dropout"]

    if mode in ["orig", "OTSeq2Set", "RL"]:
        encoder = Encoder(
            No_tokens,
            encoder_embedding_dim,
            hidden_dim,
            n_layers,
            encoder_dropout,
            rnn_type = rnn_type
        )
    elif mode == "rpw":
        encoder = ReadProcessWriteEncoder(
            No_tokens,
            encoder_embedding_dim,
            hidden_dim,
            n_layers,
            encoder_dropout,
            rnn_type=rnn_type,
            processing_steps=3,
            pad_idx=dic["pad"]
        )
    decoder = Decoder(
        No_tokens,
        decoder_embedding_dim,
        hidden_dim,
        n_layers,
        decoder_dropout,
        rnn_type = rnn_type,
        pad_idx=dic["pad"],
    )

    if args.load_embeddings:
        exported_embeddings = np.load(args.kge_parameter_location)
        import json
        with open('/content/data.json', 'r') as file:
            exported_dic = json.load(file)
        new_embedding_tensor = []
        for k, v in dic.items():
            if k in ad:
                new_embedding_tensor.append(encoder.embedding.weight.data[v])
            else:
                new_embedding_tensor.append(torch.from_numpy(exported_embeddings[exported_dic[k]]))
        if encoder.embedding.weight.data.shape == torch.stack(new_embedding_tensor).shape:
            encoder.embedding.weight.data.copy_(torch.stack(new_embedding_tensor))
            decoder.embedding.weight.data.copy_(torch.stack(new_embedding_tensor))
        else:
            assert False, "mismatching shape"
        model = Seq2Seq(encoder, decoder, device).to(device)
    else:
        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.2, 0.2)
        model = Seq2Seq(encoder, decoder, args).to(device)
        model.apply(init_weights)
    
    return model