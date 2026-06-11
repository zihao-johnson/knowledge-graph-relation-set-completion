import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder
import math
import torch
import torch.nn.functional as F
from copy import deepcopy

class Encoder(nn.Module):
    def __init__(self, len_token, embedding_dim, hidden_dim, n_layers, nhead, dropout, sequential, positional, pad_idx):
        super().__init__()
        self.n_layers = n_layers
        self.embedding = nn.Embedding(len_token, embedding_dim)
        self.embedding_dim = embedding_dim
        self.embedding.weight.data[pad_idx] = 0
        self.embedding.weight.requires_grad = False

        self.src_mask = None
        self.sequential, self.positional = sequential, positional
        if self.positional:
            self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        encoder_layers = TransformerEncoderLayer(embedding_dim, nhead, hidden_dim, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, n_layers)
        self.dropout = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    
    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            self.src_mask = self._generate_square_subsequent_mask(len(src)).to(device)
    
        # print("Mask in encoder: ", self.src_mask.shape, self.src_mask)
        embedded = self.dropout(self.embedding(src))*math.sqrt(self.embedding.embedding_dim)
        if self.positional:
            embedded = self.pos_encoder(embedded)
        if not self.sequential:
            output = self.transformer_encoder(embedded, None)
        else:
            output = self.transformer_encoder(embedded, self.src_mask)

        return [output]

class Decoder(nn.Module):
    def __init__(self, len_token, embedding_dim, hidden_dim, n_layers, nhead, dropout, sequential, positional, pad_idx):
        super().__init__()
        self.len_token = len_token
        self.n_layers = n_layers
        self.embedding = nn.Embedding(len_token, embedding_dim)
        self.embedding_dim = embedding_dim
        self.sequential, self.positional = sequential, positional
        if self.positional:
            self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        decoder_layers = TransformerDecoderLayer(embedding_dim, nhead, hidden_dim, dropout)
        self.transformer_decoder = TransformerDecoder(decoder_layers, n_layers)
        self.encoder = nn.Embedding(len_token, embedding_dim)
        # self.decoder_map = nn.Linear(embedding_dim, nhid)
        # self.decoder = nn.Linear(embedding_dim, len_token)
        self.fc_out = nn.Linear(embedding_dim, len_token)
        self.dropout = nn.Dropout(dropout)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, input, hidden):
        # input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))*math.sqrt(self.embedding.embedding_dim)
        if self.positional:
            embedded = self.pos_encoder(embedded)
        # print("embedded: ", embedded.shape)
        mask = self._generate_square_subsequent_mask(len(embedded)).to(hidden.device)
        # print("mask in decoder:", mask.shape, mask)
        # print("hidden: ", hidden.shape)
        output = self.transformer_decoder(embedded, hidden, tgt_mask = mask)
        # print("output: ", output.shape)
        prediction = self.fc_out(output)
        # print("prediction: ", prediction.shape)
        return prediction


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)

class Transformer(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert encoder.embedding_dim == decoder.embedding_dim, (
            f"Hidden dimensions of encoder and decoder must be equal!, "
            f"{encoder.embedding_dim}, {decoder.embedding_dim}"
        )
        assert encoder.n_layers == decoder.n_layers, (
            f"Encoder and decoder must have equal number of layers!, "
            f"{encoder.n_layers}, {decoder.n_layers}"
        )

    def forward(self, src, trg):
        hidden = self.encoder(src)
        output = self.decoder(trg, hidden[0])
        return output

    def train_fn(
        self,
        model,
        data_loader,
        optimizer,
        criterion,
        clip,
        teacher_forcing_ratio,
        device
    ):
        model.train()
        epoch_loss = 0

        for i, batch in enumerate(data_loader):
            src = batch["en_ids"].to(self.device)
            trg = batch["de_ids"].to(self.device)

            optimizer.zero_grad()

            output = model(src, trg[:-1])

            output_dim = output.shape[-1]
            output = output.reshape(-1, output_dim)
            trg_gold = trg[1:].reshape(-1)

            loss = criterion(output, trg_gold)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

    def evaluate_fn(self, model, data_loader, criterion, device):
        model.eval()
        epoch_loss = 0

        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                src = batch["en_ids"].to(self.device)
                trg = batch["de_ids"].to(self.device)

                output = model(src, trg[:-1])

                output_dim = output.shape[-1]
                output = output.reshape(-1, output_dim)
                trg_gold = trg[1:].reshape(-1)

                loss = criterion(output, trg_gold)
                epoch_loss += loss.item()

        return epoch_loss / len(data_loader)

    def translate_sentence(
        self,
        sentence,
        model,
        dic,
        sos_token,
        eos_token,
        device,
        max_output_length=15,
        allow_repeated=True
    ):
        model.eval()

        with torch.no_grad():
            tensor = torch.LongTensor(sentence).unsqueeze(-1).to(self.device)
            hidden = model.encoder(tensor)

            inputs = [sos_token]

            for _ in range(max_output_length):
                inputs_tensor = torch.LongTensor([inputs]).to(self.device).T
                output = model.decoder(inputs_tensor, hidden[0])

                next_token_logits = output[-1, 0].clone()
                predicted_token = next_token_logits.argmax(-1).item()

                if not allow_repeated:
                    tries = 0
                    while predicted_token in inputs + sentence:
                        next_token_logits[predicted_token] = float("-inf")
                        predicted_token = next_token_logits.argmax(-1).item()
                        tries += 1
                        if tries == 10:
                            break
                else:
                    tries = 0
                    while predicted_token in sentence:
                        next_token_logits[predicted_token] = float("-inf")
                        predicted_token = next_token_logits.argmax(-1).item()
                        tries += 1
                        if tries == 10:
                            break

                inputs.append(predicted_token)

                if predicted_token == eos_token:
                    break

        return inputs

def create_transformer(args, len_token, dic, device):
    defaults = {
    "embedding_dim": 256,
    "hidden_dim": 256,
    "n_layers": 2,
    "encoder_dropout" : 0.5,
    "decoder_dropout" : 0.5,
    "nhead" : 4,
    "sequential": False,
     "positional": False
    }

    model_cfg = {**defaults, **getattr(args, "model_config", {})}

    embedding_dim = model_cfg["embedding_dim"]
    hidden_dim = model_cfg["hidden_dim"]
    n_layers = model_cfg["n_layers"]
    encoder_dropout = model_cfg["encoder_dropout"]
    decoder_dropout = model_cfg["decoder_dropout"]
    nhead = model_cfg["nhead"]
    sequential, positional = model_cfg["sequential"], model_cfg["positional"]
    
    encoder = Encoder(
        len_token,
        embedding_dim,
        hidden_dim,
        n_layers,
        nhead,
        encoder_dropout,
        sequential,
        positional,
        pad_idx
    )
    decoder = Decoder(
        len_token,
        embedding_dim,
        hidden_dim,
        n_layers,
        nhead,
        decoder_dropout,
        positional,
        pad_idx
    )

    if args.load_embeddings:
        exported_embeddings = np.load("/content/embeddings.npy")

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

    model = Transformer(encoder, decoder, device).to(device)

    if not args.load_embeddings:
        def init_weights(m):
            for name, param in m.named_parameters():
                nn.init.uniform_(param.data, -0.1, 0.1)
        model.apply(init_weights)
        model.encoder.embedding.weight.data[dic["pad"]] = 0
        model.encoder.embedding.weight.requires_grad = False
        model.decoder.embedding.weight.data[dic["pad"]] = 0
        model.decoder.embedding.weight.requires_grad = False
    return model