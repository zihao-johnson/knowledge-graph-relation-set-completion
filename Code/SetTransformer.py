import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RowFF(nn.Module):
    """
    Row-wise feed-forward network.
    Applies the same MLP independently to each set element.
    Input:  (L, B, D)
    Output: (L, B, D)
    """
    def __init__(self, dim, hidden_dim, dropout=0.0, activation="relu"):
        super().__init__()

        if activation == "relu":
            act = nn.ReLU()
        elif activation == "gelu":
            act = nn.GELU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            raise ValueError(f"Unsupported activation: {activation}")

        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            act,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class MAB(nn.Module):
    """
    Multihead Attention Block from Set Transformer.

    MAB(X, Y):
        H = LN(X + Multihead(X, Y, Y))
        O = LN(H + rFF(H))

    Shapes:
        X: (Lx, B, D)
        Y: (Ly, B, D)
        out: (Lx, B, D)
    """
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.0, activation="relu"):
        super().__init__()
        self.mha = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=False
        )
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.rff = RowFF(dim, hidden_dim, dropout=dropout, activation=activation)

    def forward(self, X, Y, key_padding_mask=None):
        """
        Args:
            X: (Lx, B, D) query set
            Y: (Ly, B, D) key/value set
            key_padding_mask: (B, Ly), True where padded

        Returns:
            out: (Lx, B, D)
        """
        attn_out, _ = self.mha(
            query=X,
            key=Y,
            value=Y,
            key_padding_mask=key_padding_mask
        )
        H = self.ln1(X + attn_out)
        out = self.ln2(H + self.rff(H))
        return out


class SAB(nn.Module):
    """
    Set Attention Block:
        SAB(X) = MAB(X, X)
    """
    def __init__(self, dim, num_heads, hidden_dim, dropout=0.0, activation="relu"):
        super().__init__()
        self.mab = MAB(dim, num_heads, hidden_dim, dropout=dropout, activation=activation)

    def forward(self, X, key_padding_mask=None):
        return self.mab(X, X, key_padding_mask=key_padding_mask)


class ISAB(nn.Module):
    """
    Induced Set Attention Block:
        H = MAB(I, X)
        O = MAB(X, H)

    I is a learned set of inducing points of size m.

    Shapes:
        X: (L, B, D)
        H: (m, B, D)
        O: (L, B, D)
    """
    def __init__(
        self,
        dim,
        num_heads,
        hidden_dim,
        num_inducing_points,
        dropout=0.0,
        activation="relu"
    ):
        super().__init__()
        self.inducing_points = nn.Parameter(torch.randn(num_inducing_points, 1, dim))

        self.mab1 = MAB(dim, num_heads, hidden_dim, dropout=dropout, activation=activation)
        self.mab2 = MAB(dim, num_heads, hidden_dim, dropout=dropout, activation=activation)

    def forward(self, X, key_padding_mask=None):
        """
        Args:
            X: (L, B, D)
            key_padding_mask: (B, L)

        Returns:
            out: (L, B, D)
        """
        B = X.size(1)
        I = self.inducing_points.expand(-1, B, -1)   # (m, B, D)

        # H = MAB(I, X)
        H = self.mab1(I, X, key_padding_mask=key_padding_mask)

        # O = MAB(X, H)
        # H has no padding, so key_padding_mask=None
        out = self.mab2(X, H, key_padding_mask=None)
        return out


class PMA(nn.Module):
    """
    Pooling by Multihead Attention.

    PMA_k(Z) = MAB(S, rFF(Z))

    S is a learned set of seed vectors of size k.

    Input:
        Z: (L, B, D)

    Output:
        pooled: (k, B, D)
    """
    def __init__(
        self,
        dim,
        num_heads,
        hidden_dim,
        num_seeds=1,
        dropout=0.0,
        activation="relu"
    ):
        super().__init__()
        self.num_seeds = num_seeds
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, 1, dim))
        self.pre_rff = RowFF(dim, hidden_dim, dropout=dropout, activation=activation)
        self.mab = MAB(dim, num_heads, hidden_dim, dropout=dropout, activation=activation)

    def forward(self, Z, key_padding_mask=None):
        """
        Args:
            Z: (L, B, D)
            key_padding_mask: (B, L)

        Returns:
            pooled: (k, B, D)
        """
        B = Z.size(1)
        S = self.seed_vectors.expand(-1, B, -1)   # (k, B, D)
        Z_ff = self.pre_rff(Z)
        pooled = self.mab(S, Z_ff, key_padding_mask=key_padding_mask)
        return pooled


class SetTransformerEncoder(nn.Module):
    """
    Proper Set Transformer encoder:
        embedding -> stack of SAB or ISAB

    Input:
        src: (L, B) LongTensor token ids

    Output:
        Z: (L, B, D)
        src_key_padding_mask: (B, L)
    """
    def __init__(
        self,
        len_token,
        embedding_dim,
        hidden_dim,
        n_layers,
        nhead,
        dropout,
        pad_idx,
        use_isab=False,
        num_inducing_points=16,
        activation="relu",
    ):
        super().__init__()

        self.len_token = len_token
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.nhead = nhead
        self.pad_idx = pad_idx
        self.use_isab = use_isab
        
        self.embedding = nn.Embedding(
            len_token,
            embedding_dim,
            padding_idx=pad_idx
        )
        self.dropout = nn.Dropout(dropout)

        with torch.no_grad():
            self.embedding.weight[pad_idx].zero_()

        blocks = []
        for _ in range(n_layers):
            if use_isab:
                blocks.append(
                    ISAB(
                        dim=embedding_dim,
                        num_heads=nhead,
                        hidden_dim=hidden_dim,
                        num_inducing_points=num_inducing_points,
                        dropout=dropout,
                        activation=activation,
                    )
                )
            else:
                blocks.append(
                    SAB(
                        dim=embedding_dim,
                        num_heads=nhead,
                        hidden_dim=hidden_dim,
                        dropout=dropout,
                        activation=activation,
                    )
                )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, src):
        """
        Args:
            src: (L, B)

        Returns:
            Z: (L, B, D)
            src_key_padding_mask: (B, L)
        """
        src_key_padding_mask = (src.transpose(0, 1) == self.pad_idx)  # (B, L)

        X = self.dropout(self.embedding(src)) * math.sqrt(self.embedding_dim)  # (L,B,D)

        for block in self.blocks:
            X = block(X, key_padding_mask=src_key_padding_mask)

        return X, src_key_padding_mask


class SetTransformerDecoder(nn.Module):
    """
    Proper Set Transformer decoder:
        PMA -> optional SAB stack -> optional final projection

    If num_outputs == 1:
        returns pooled representation of shape (B, D)

    If num_outputs > 1:
        returns pooled set of shape (k, B, D)
    """
    def __init__(
        self,
        embedding_dim,
        hidden_dim,
        nhead,
        dropout=0.0,
        num_outputs=1,
        n_decoder_sab=0,
        activation="relu",
        output_dim=None,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.num_outputs = num_outputs
        self.n_decoder_sab = n_decoder_sab
        self.output_dim = output_dim

        self.pma = PMA(
            dim=embedding_dim,
            num_heads=nhead,
            hidden_dim=hidden_dim,
            num_seeds=num_outputs,
            dropout=dropout,
            activation=activation,
        )

        self.sab_blocks = nn.ModuleList([
            SAB(
                dim=embedding_dim,
                num_heads=nhead,
                hidden_dim=hidden_dim,
                dropout=dropout,
                activation=activation,
            )
            for _ in range(n_decoder_sab)
        ])

        self.output_proj = None
        if output_dim is not None and output_dim != embedding_dim:
            self.output_proj = nn.Linear(embedding_dim, output_dim)

    def forward(self, Z, key_padding_mask=None):
        """
        Args:
            Z: (L, B, D)
            key_padding_mask: (B, L)

        Returns:
            if num_outputs == 1:
                out: (B, D_or_output_dim)
            else:
                out: (k, B, D_or_output_dim)
        """
        H = self.pma(Z, key_padding_mask=key_padding_mask)  # (k, B, D)

        for sab in self.sab_blocks:
            H = sab(H, key_padding_mask=None)

        if self.output_proj is not None:
            H = self.output_proj(H)

        if self.num_outputs == 1:
            return H.squeeze(0)  # (B, D)
        return H  # (k, B, D)


class SetTransformer(nn.Module):
    """
    Full Set Transformer:
        Encoder(X) -> Z
        Decoder(Z) -> pooled output

    Input:
        src: (L, B)

    Output:
        if num_outputs == 1:
            (B, D)
        else:
            (k, B, D)
    """
    def __init__(
        self,
        len_token,
        pad_idx,
        args,
        use_isab=False,
        num_inducing_points=16,
        num_outputs=1,
        n_decoder_sab=0,
        activation="relu",
        output_dim=None,
    ):
        super().__init__()
        self.len_token = len_token
        self.embedding_dim = args.embedding_dim
        self.hidden_dim = args.hidden_dim
        self.n_layers = args.n_layers
        self.nhead = args.nhead
        self.dropout = args.dropout
        self.pad_idx = pad_idx
        self.args = args
        self.encoder = SetTransformerEncoder(
            len_token=len_token,
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            n_layers=self.n_layers,
            nhead=self.nhead,
            dropout=self.dropout,
            pad_idx=self.pad_idx,
            use_isab=use_isab,
            num_inducing_points=num_inducing_points,
            activation=activation,
        )

        self.decoder = SetTransformerDecoder(
            embedding_dim=self.embedding_dim,
            hidden_dim=self.hidden_dim,
            nhead=self.nhead,
            dropout=self.dropout,
            num_outputs=num_outputs,
            n_decoder_sab=n_decoder_sab,
            activation=activation,
            output_dim=output_dim,
        )

    def forward(self, src, return_set_encoding=False):
        """
        Args:
            src: (L, B)
            return_set_encoding: if True, also return encoder output Z

        Returns:
            pooled
            or (pooled, Z, mask)
        """
        Z, mask = self.encoder(src)
        pooled = self.decoder(Z, key_padding_mask=mask)

        if return_set_encoding:
            return [Z, mask, pooled]
        return pooled

    @property
    def embedding(self):
        return self.encoder.embedding