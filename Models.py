import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import TransformerEncoder, TransformerEncoderLayer

# --------------------------------
#       RelSetE  Encoders        |
# --------------------------------
class RelSetE_Encoder(nn.Module):
    def __init__(self, len_token, pad_idx, args):
        super().__init__()
        self.n_layers = args.n_layers
        self.embedding_dim = args.embedding_dim
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(len_token, self.embedding_dim, padding_idx=pad_idx)
        self.embedding.weight.data[pad_idx] = 0
        # self.embedding.weight.requires_grad = False

        self.src_mask = None
        
        encoder_layers = TransformerEncoderLayer(self.embedding_dim, args.nhead, args.hidden_dim, args.dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, self.n_layers)
        self.dropout_layer = nn.Dropout(args.dropout)

        # ---- Attention Pooling (PMA / seed pooling) ----
        self.pool_attn = nn.MultiheadAttention(embed_dim=self.embedding_dim, num_heads=args.nhead, dropout=args.dropout)
        self.pool_seed = nn.Parameter(torch.randn(1, 1, self.embedding_dim))  # (1,1,d)

    def _generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

    def forward(self, src):
        if self.src_mask is None or self.src_mask.size(0) != len(src):
            device = src.device
            self.src_mask = self._generate_square_subsequent_mask(len(src)).to(device)

        src_key_padding_mask = (src.transpose(0, 1) == self.pad_idx)

        embedded = self.dropout_layer(self.embedding(src))*math.sqrt(self.embedding.embedding_dim)

        embedded = F.layer_norm(embedded, embedded.shape[-1:])
        
        output = self.transformer_encoder(
            embedded,
            src_key_padding_mask=src_key_padding_mask
        )
 
        # --- Attention Pooling (PMA) ---
        # query: (1,B,d)
        B = src.shape[1]
        q = self.pool_seed.expand(1, B, self.embedding_dim).contiguous()
        pooled, _ = self.pool_attn(
            query=q,
            key=output,
            value=output,
            key_padding_mask=src_key_padding_mask  # (B,L)
        )  # (1,B,d)

        z = pooled.squeeze(0)  # (B,d)
        return z


# --------------------------------
#          MLP  Encoders         |
# --------------------------------
class FlattenMLPEncoder(nn.Module):
    """
    MLP encoder: batch of token ID sequences -> flattened embedding -> MLP -> representation z

    Inputs:
        x_ids: LongTensor (L, B) padded with pad_idx
    Outputs:
        z: Tensor (B, out_dim)

    Note:
        - This encoder is order-sensitive because it flattens the sequence.
        - It requires a fixed maximum sequence length `max_len`.
        - Input is sequence-first: (L, B)
    """
    def __init__(
        self,
        len_token,
        pad_idx,
        args,
        freeze_embedding = False
    ):
        super().__init__()
        self.pad_idx = pad_idx
        self.embedding_dim = args.embedding_dim
        self.max_len = args.max_len
        self.hidden_dim = args.hidden_dim
        self.embedding = nn.Embedding(
            len_token,
            self.embedding_dim,
            padding_idx=pad_idx
        )
        self.args = args

        if freeze_embedding:
            self.embedding.weight.requires_grad = False

        input_dim = self.max_len * self.embedding_dim
        out_dim = self.embedding_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(self.args.dropout),

            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Sigmoid(),
            nn.Dropout(self.args.dropout),

            nn.Linear(self.hidden_dim, out_dim),
            nn.LogSoftmax(),
        )

    def forward(self, x_ids: torch.Tensor) -> torch.Tensor:
        """
        x_ids: (L, B)
        returns: (B, out_dim)
        """
        if x_ids.dim() != 2:
            raise ValueError(f"x_ids must have shape (L, B), got {x_ids.shape}")

        L, B = x_ids.shape

        if L > self.max_len:
            raise ValueError(
                f"Input sequence length {L} exceeds max_len={self.max_len}"
            )

        # Convert from (L, B) -> (B, L)
        x_ids = x_ids.transpose(0, 1)

        # Right-pad up to max_len if needed
        if L < self.max_len:
            pad_size = self.max_len - L
            pad = torch.full(
                (B, pad_size),
                self.pad_idx,
                dtype=x_ids.dtype,
                device=x_ids.device
            )
            x_ids = torch.cat([x_ids, pad], dim=1)

        # (B, max_len, embedding_dim)
        x = self.embedding(x_ids)

        # Zero out padded embeddings explicitly
        # mask = (x_ids != self.pad_idx).unsqueeze(-1).float()
        # x = x * mask

        # (B, max_len * embedding_dim)
        x = x.reshape(B, self.max_len * self.embedding_dim)

        # (B, out_dim)
        z = self.mlp(x)
        return z


# --------------------------------
#       DeepSet  Encoders        |
# --------------------------------
class DeepSet(nn.Module):
    """
    Permutation-equivariant layer for set inputs.

    Input:
        x: LongTensor of shape (L, B) or (B, L)
           token ids with padding

    Output:
        Tensor of shape (B, d)

    Implements an equivariant transform on element embeddings:
        h_i = act(lambda * x_i + gamma * pool({x_j}))
    then aggregates across the set:
        z = sum_i h_i   or   mean_i h_i
    """
    def __init__(
        self,
        len_token,
        pad_idx,
        args,
        activation="relu",
        pool="mean",
        per_dim=False,
        input_format="LB",   # "LB" means (L, B), "BL" means (B, L)
    ):
        super().__init__()

        assert pool in ("sum", "mean")
        assert input_format in ("LB", "BL")

        self.len_token = len_token
        self.args = args
        self.embedding_dim = args.embedding_dim
        self.pad_idx = pad_idx
        self.pool = pool
        self.input_format = input_format
        self.dropout_layer = nn.Dropout(args.dropout)
        self.embedding = nn.Embedding(
            self.len_token,
            self.embedding_dim,
            padding_idx=pad_idx
        )

        if per_dim:
            self.lam = nn.Parameter(torch.ones(1, 1, self.embedding_dim))
            self.gam = nn.Parameter(torch.ones(1, 1, self.embedding_dim))
        else:
            self.lam = nn.Parameter(torch.tensor(1.0))
            self.gam = nn.Parameter(torch.tensor(1.0))

        if activation == "relu":
            self.act = F.relu
        elif activation == "tanh":
            self.act = torch.tanh
        elif activation == "sigmoid":
            self.act = torch.sigmoid
        elif activation == "gelu":
            self.act = F.gelu
        elif activation == "none":
            self.act = lambda t: t
        else:
            raise ValueError(f"Unknown activation: {activation}")

    def forward(self, x):
        # Accept either (L, B) or (B, L), normalize to (B, L)
        if self.input_format == "LB":
            x = x.transpose(0, 1)
        # now x is (B, L)

        mask = (x != self.pad_idx)                      # (B, L)
        emb = self.dropout_layer(self.embedding(x))*math.sqrt(self.embedding.embedding_dim)# (B, L, d)

        # Ignore padded positions for max-pooling
        # emb_masked = emb.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        # pooled = emb_masked.max(dim=1, keepdim=True).values   # (B, 1, d)

        # If a whole row is padding, max becomes -inf; replace with zeros
        # pooled = torch.where(torch.isfinite(emb), emb, torch.zeros_like(emb))

        # Equivariant elementwise transform
        out = self.lam * emb + self.gam * emb        # (B, L, d)
        # out = self.act(out)

        # Zero padded positions before final aggregation
        out = out * mask.unsqueeze(-1).float()

        pooled_out = out.sum(dim=1)                     # (B, d)
        if self.pool == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1).float()
            pooled_out = pooled_out / denom

        return pooled_out


# --------------------------------
#              Decoders          |
# --------------------------------
class Scoring_Decoder(nn.Module):
    def __init__(self, len_token, pad_idx, args):
        super().__init__()
        self.len_token = len_token
        self.embedding_dim = args.embedding_dim
        
        self.embedding = nn.Embedding(len_token, self.embedding_dim, padding_idx=pad_idx)
        
        # self.decoder_layers = TransformerDecoderLayer(self.embedding_dim, args.nhead, args.hidden_dim, args.dropout)
    
    def get_relation_embedding(self):
        emb = self.embedding.weight*math.sqrt(self.embedding.embedding_dim)
        return emb

    def forward(self, pos_sample, neg_sample, hidden):
        """
        pos_sample: LongTensor (B, P)
        neg_sample: LongTensor (B, K)
        hidden    : Tensor     (B, d)   # pooled query embedding
        """

        # 1. embedding lookup ONLY
        pos_emb = self.embedding(pos_sample)*math.sqrt(self.embedding.embedding_dim)
        neg_emb = self.embedding(neg_sample)*math.sqrt(self.embedding.embedding_dim)
        # 2. dot-product scoring
        # (1,B,d) * (P,B,d) -> (B,P)
        # print("\n")
        # print(hidden.shape, pos_emb.shape,neg_emb.shape,"\n")

        # Normalize:
        hidden = F.normalize(hidden, dim=-1)
        pos_emb = F.normalize(pos_emb, dim=-1)
        neg_emb = F.normalize(neg_emb, dim=-1)

        pos_score = (hidden.unsqueeze(0) * pos_emb).sum(dim=-1)

        # (B,1,d) * (B,K,d) -> (B,K)
        neg_score = (hidden.unsqueeze(0) * neg_emb).sum(dim=-1)

        return pos_score, neg_score


class MLC_Decoder(nn.Module):
    def __init__(self, len_token, pad_idx, args):
        super().__init__()
        self.len_token = len_token
        self.embedding_dim = args.embedding_dim
        self.dropout = args.dropout

        self.dropout = nn.Dropout(self.dropout)
        self.MLC_out_layer = nn.Sequential(nn.Linear(self.embedding_dim,512),
                                                    nn.Sigmoid(),
                                                    nn.Linear(512,256),
                                                    nn.Sigmoid(),
                                                    nn.Linear(256,self.len_token),
                                                    )

    def forward(self, hidden):
        prediction = self.dropout(self.MLC_out_layer(hidden))
        return prediction


# --------------------------------
#          Full frameworks       |
# --------------------------------
class Score_base(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        # assert (
        #     encoder.embedding_dim == decoder.embedding_dim
        # ), "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src, pos_sample, neg_sample):
        hidden = self.encoder(src)
        pos_score, neg_score = self.decoder(pos_sample, neg_sample, hidden)
        return pos_score, neg_score

    def rank_topk(self, input_ids, pad_idx, k):
        hidden = self.encoder(input_ids)

        rel_emb_weight = self.decoder.get_relation_embedding()

        # Normalize:
        rel_emb_weight = F.normalize(rel_emb_weight, dim=-1)
        hidden = F.normalize(hidden, dim=-1)

        scores = hidden @ rel_emb_weight.t()
        # mask out input relations (do not recommend seen relations)
        L, B = input_ids.shape
        valid = input_ids != pad_idx
        rows = torch.arange(B, device=hidden.device).unsqueeze(1).expand(B, L)
        
        # algin dimensions:
        scores[rows[valid.T], input_ids.T[valid.T]] = float("-inf")

        topk_scores, topk_ids = torch.topk(scores, k=k, dim=1)
        return topk_ids, topk_scores


class MLC_base(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert (
            encoder.embedding_dim == decoder.embedding_dim
        ), "Hidden dimensions of encoder and decoder must be equal!"

    def forward(self, src):
        z = self.encoder(src)
        output = self.decoder(z)
        return output
    
    def rank_topk(self, input_ids, pad_idx, k):
        scores = self.forward(input_ids)
        # print(scores.shape)
        
        # mask out input relations (do not recommend seen relations)
        L, B = input_ids.shape
        valid = input_ids != pad_idx
        rows = torch.arange(B, device=self.device).unsqueeze(1).expand(B, L)
        
        # algin dimensions:
        scores[rows[valid.T], input_ids.T[valid.T]] = float("-inf")

        topk_scores, topk_ids = torch.topk(scores, k=k, dim=1)
        return topk_ids, topk_scores