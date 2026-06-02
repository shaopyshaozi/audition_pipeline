from typing import *
import numpy as np
import torch
import torch.nn as nn
from models.arch.base.norm import *
from models.arch.base.non_linear import *
from models.arch.base.linear_group import LinearGroup

# # just for test the net
# from base.norm import *
# from base.non_linear import *
# from base.linear_group import LinearGroup

from torch import Tensor
from torch.nn import MultiheadAttention
import math


def one_hot_positional_encoding(width: int, width_emb_dim: int = 3, width_stage: int = 15) -> float:
    w = width/width_stage -1 
    return torch.nn.functional.one_hot(w.to(torch.int64), width_emb_dim).float().squeeze(dim=0)


def cyclic_positional_encoding(phi: int, D: int = 40, alpha: int = 20) -> float:
    """
    Generates a cyclic positional encoding matrix for a sequence of a given maximum length and model dimension.
    
    Parameters:
    phi(int): The DOA of the signal
    D (int): The dimensionality of the model.
    alpha (int): A scaling factor. Default is 20.
    
    """
    phi_rad = torch.tensor(math.radians(phi))

    # Create a range of dimensions
    pe = torch.zeros(D)

    # Calculate the encoding using sine and cosine functions 
    for j in range(D // 2):
        angle = alpha / (10000 ** (2 * j / D))
        pe[2 * j] = torch.sin(torch.sin(phi_rad) * angle)
        pe[2 * j + 1] = torch.sin(torch.cos(phi_rad) * angle)
    return pe


class ClueEncoder(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super(ClueEncoder, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.LN = nn.LayerNorm(output_dim)
        self.PReLU = nn.PReLU()

    def forward(self, pe):
        # pe shape: [B, T, D]
        B, T, D = pe.size()
        pe = pe.view(-1, D)  # Reshape to (B*T, D)
        pe = self.linear(pe)  
        pe = self.LN(pe)  
        pe = self.PReLU(pe)  
        pe = pe.view(B, T, -1)  # Reshape back to (B, T, C)
        pe = pe.unsqueeze(-1)  # Add a new dimension to get (B, T, C, 1)
        pe = pe.permute(0, 3, 1, 2) #(B,1,T,C)
        return pe


class WidthProjection(nn.Module):
    def __init__(self, width_dim: int = 3, hidden_dim: int = 64, channels: int = 192):
        super().__init__()

        self.fc = nn.Linear(width_dim, hidden_dim)
        self.conv = nn.Conv2d(hidden_dim, channels, kernel_size=(1, 1))  # 1x1 Conv

        self.weight = nn.Parameter(torch.tensor(1.0))  

        # self.LN = nn.LayerNorm( channels )
        # self.act = nn.Tanh()


    def forward(self, w_emb, x):
        """
        width: [B, 3]
        x: [B, F, T, C]
        """
        B, F, T, C = x.shape
        
        w_projected = self.fc(w_emb)  # [B, H]
        w_projected = self.conv(w_projected.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, F, T))  # [B, H, F, T]

        # mask = self.act( self.LN(w_projected.permute(0, 2, 3, 1) )) # [B, F, T, C]   
        # return x + x * mask

        mask = w_projected.permute(0, 2, 3, 1)  # [B, F, T, C]
        return x + self.weight * x * mask





class SpatialNetLayer(nn.Module):

    def __init__(
            self,
            dim_hidden: int,
            dim_ffn: int,
            dim_squeeze: int,
            num_freqs: int,
            num_heads: int,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full: nn.Module = None,
    ) -> None:
        super().__init__()
        f_conv_groups = conv_groups[0]
        t_conv_groups = conv_groups[1]
        f_kernel_size = kernel_size[0]
        t_kernel_size = kernel_size[1]

        # cross-band block
        # frequency-convolutional module
        self.fconv1 = nn.ModuleList([
            new_norm(norms[3], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])
        # full-band linear module
        self.norm_full = new_norm(norms[5], dim_hidden, seq_last=False, group_size=None, num_groups=f_conv_groups)
        self.full_share = False if full == None else True
        self.squeeze = nn.Sequential(nn.Conv1d(in_channels=dim_hidden, out_channels=dim_squeeze, kernel_size=1), nn.SiLU())
        self.dropout_full = nn.Dropout2d(dropout[2]) if dropout[2] > 0 else None
        self.full = LinearGroup(num_freqs, num_freqs, num_groups=dim_squeeze) if full == None else full
        self.unsqueeze = nn.Sequential(nn.Conv1d(in_channels=dim_squeeze, out_channels=dim_hidden, kernel_size=1), nn.SiLU())
        # frequency-convolutional module
        self.fconv2 = nn.ModuleList([
            new_norm(norms[4], dim_hidden, seq_last=True, group_size=None, num_groups=f_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_hidden, kernel_size=f_kernel_size, groups=f_conv_groups, padding='same', padding_mode=padding),
            nn.PReLU(dim_hidden),
        ])

        # narrow-band block
        # MHSA module
        self.norm_mhsa = new_norm(norms[0], dim_hidden, seq_last=False, group_size=None, num_groups=t_conv_groups)
        self.mhsa = MultiheadAttention(embed_dim=dim_hidden, num_heads=num_heads, batch_first=True)
        self.dropout_mhsa = nn.Dropout(dropout[0])
        # T-ConvFFN module
        self.tconvffn = nn.ModuleList([
            new_norm(norms[1], dim_hidden, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.Conv1d(in_channels=dim_hidden, out_channels=dim_ffn, kernel_size=1),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            new_norm(norms[2], dim_ffn, seq_last=True, group_size=None, num_groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_ffn, kernel_size=t_kernel_size, padding='same', groups=t_conv_groups),
            nn.SiLU(),
            nn.Conv1d(in_channels=dim_ffn, out_channels=dim_hidden, kernel_size=1),
        ])
        self.dropout_tconvffn = nn.Dropout(dropout[1])

    def forward(self, x: Tensor, att_mask: Optional[Tensor] = None) -> Tensor:
        r"""
        Args:
            x: shape [B, F, T, H]
            att_mask: the mask for attention along T. shape [B, T, T]

        Shape:
            out: shape [B, F, T, H]
        """
        x = x + self._fconv(self.fconv1, x)
        x = x + self._full(x)
        x = x + self._fconv(self.fconv2, x)
        x_, attn = self._tsa(x, att_mask)
        x = x + x_
        x = x + self._tconvffn(x)
        return x, attn

    def _tsa(self, x: Tensor, attn_mask: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
        B, F, T, H = x.shape
        x = self.norm_mhsa(x)
        x = x.reshape(B * F, T, H)
        need_weights = False if hasattr(self, "need_weights") else self.need_weights
        x, attn = self.mhsa.forward(x, x, x, need_weights=need_weights, average_attn_weights=False, attn_mask=attn_mask)
        x = x.reshape(B, F, T, H)
        return self.dropout_mhsa(x), attn

    def _tconvffn(self, x: Tensor) -> Tensor:
        B, F, T, H0 = x.shape
        # T-Conv
        x = x.transpose(-1, -2)  # [B,F,H,T]
        x = x.reshape(B * F, H0, T)
        for m in self.tconvffn:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=F)
            else:
                x = m(x)
        x = x.reshape(B, F, H0, T)
        x = x.transpose(-1, -2)  # [B,F,T,H]
        return self.dropout_tconvffn(x)

    def _fconv(self, ml: nn.ModuleList, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        for m in ml:
            if type(m) == GroupBatchNorm:
                x = m(x, group_size=T)
            else:
                x = m(x)
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def _full(self, x: Tensor) -> Tensor:
        B, F, T, H = x.shape
        x = self.norm_full(x)
        x = x.permute(0, 2, 3, 1)  # [B,T,H,F]
        x = x.reshape(B * T, H, F)
        x = self.squeeze(x)  # [B*T,H',F]
        if self.dropout_full:
            x = x.reshape(B, T, -1, F)
            x = x.transpose(1, 3)  # [B,F,H',T]
            x = self.dropout_full(x)  # dropout some frequencies in one utterance
            x = x.transpose(1, 3)  # [B,T,H',F]
            x = x.reshape(B * T, -1, F)

        x = self.full(x)  # [B*T,H',F]
        x = self.unsqueeze(x)  # [B*T,H,F]
        x = x.reshape(B, T, H, F)
        x = x.permute(0, 3, 1, 2)  # [B,F,T,H]
        return x

    def extra_repr(self) -> str:
        return f"full_share={self.full_share}"


class DSENet(nn.Module):

    def __init__(
            self,
            dim_input: int,  # the input dim for each time-frequency point
            dim_output: int,  # the output dim for each time-frequency point
            dim_squeeze: int,
            num_layers: int,
            num_freqs: int,
            encoder_kernel_size: int = 5,
            dim_hidden: int = 192,
            dim_ffn: int = 384,
            num_heads: int = 2,
            dropout: Tuple[float, float, float] = (0, 0, 0),
            kernel_size: Tuple[int, int] = (5, 3),
            conv_groups: Tuple[int, int] = (8, 8),
            norms: List[str] = ("LN", "LN", "GN", "LN", "LN", "LN"),
            padding: str = 'zeros',
            full_share: int = 0,  # share from layer 0
            d_embedding: int = 40,
            d_alpha: int = 20,
            width_emb_dim: int = 3,
            width_stage: int = 15,
            width_control: bool = False,
    ):
        super().__init__()

        self.d_embedding = d_embedding
        self.d_alpha = d_alpha
        self.width_emb_dim = width_emb_dim
        self.width_stage = width_stage


        # encoder
        self.encoder = nn.Conv1d(in_channels=dim_input, out_channels=dim_hidden, kernel_size=encoder_kernel_size, stride=1, padding="same")

  
        # spatialnet layers
        full = None
        layers = []
        width_layers = []

        for l in range(num_layers):
            clue_layer = ClueEncoder(input_dim = d_embedding, output_dim=dim_hidden)
            # layernorm = nn.LayerNorm(normalized_shape)
            layer = SpatialNetLayer(
                dim_hidden=dim_hidden,
                dim_ffn=dim_ffn,
                dim_squeeze=dim_squeeze,
                num_freqs=num_freqs,
                num_heads=num_heads,
                dropout=dropout,
                kernel_size=kernel_size,
                conv_groups=conv_groups,
                norms=norms,
                padding=padding,
                full=full if l > full_share else None,
            )
            if hasattr(layer, 'full'):
                full = layer.full
            layers.append(clue_layer)
            layers.append(layer)

            w_pro = WidthProjection()
            width_layers.append(w_pro)

        self.layers = nn.ModuleList(layers)

        self.width_control = width_control
        if width_control:
            self.width_layers = nn.ModuleList(width_layers)


        # decoder
        self.decoder = nn.Linear(in_features=dim_hidden, out_features=dim_output)





    def forward(self, x: Tensor, DOA: Tensor, width: Tensor, return_attn_score: bool = False) -> Tensor:
        # x: [Batch, Freq, Time, Feature]
        # DOA：[Batch, 1] width: [Batch, 1]

        B, F, T, H0 = x.shape
        x = self.encoder(x.reshape(B * F, T, H0).permute(0, 2, 1)).permute(0, 2, 1)
        H = x.shape[2]


        # DOA embedding
        cyc_pos = list(map(lambda doa: cyclic_positional_encoding(doa, self.d_embedding, self.d_alpha), DOA)) #[B,D]
        pe = torch.stack(cyc_pos).to(device = DOA.device) # [B,D]
        pe = pe.unsqueeze(1).repeat(1, T, 1) # [B,T,D]

        # width embedding
        width_emb = list(map(lambda w: one_hot_positional_encoding(w, self.width_emb_dim, self.width_stage), width)) # [B,3]
        w_emb = torch.stack(width_emb).to(device = DOA.device) # [B,3]

        attns = [] if return_attn_score else None
        x = x.reshape(B, F, T, H)
        for c,m,w in zip(self.layers[::2],self.layers[1::2],self.width_layers):
            setattr(m, "need_weights", return_attn_score)
            # The embedding is multiplied element-wise with the output of the encoder and 
            # SpatialNetLayer except for the final layer, across the channel and the time dimensions.

            clue = c(pe) # [B,1,T,C]
            assert H == clue.shape[-1]

            x = x * clue #[B,F,T,C]

            if self.width_control:
                x = w(w_emb, x)

            x, attn = m(x)
            if return_attn_score:
                attns.append(attn)

        y = self.decoder(x)


        if return_attn_score:
            return y.contiguous(), attns
        else:
            return y.contiguous()


if __name__ == '__main__':
    x = torch.randn((1, 129, 251, 4))#.cuda() # [B,F,T,2C]  251 = 8 kHz; 501 = 16 kHz
    doa = torch.randint(0,360,(1,1))#.cuda()
    w = torch.randint(15,45,(1,1))#.cuda()
    net_small = DSENet(
        dim_input=4,
        dim_output=4,
        num_layers=8,
        dim_hidden=192,
        dim_ffn=192,
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        dim_squeeze=8,
        num_freqs=129,
        full_share=0,
    )#.cuda()
    # from packaging.version import Version
    # if Version(torch.__version__) >= Version('2.0.0'):
    #     SSFNet_small = torch.compile(SSFNet_small)
    # torch.cuda.synchronize(7)

    import time
    ts = time.time()
    y = net_small(x,doa,w)
    # torch.cuda.synchronize(7)
    te = time.time()
    print(net_small)
    print(y.shape)
    print(te - ts)

    # net_small = net_small.to('meta')
    # x = x.to('meta')
    # doa = doa.to('meta')
    from torch.utils.flop_counter import FlopCounterMode # requires torch>=2.1.0
    with FlopCounterMode(net_small, display=False) as fcm:
        y = net_small(x,doa,w)
        flops_forward_eval = fcm.get_total_flops()
        res = y.sum()
        res.backward()
        flops_backward_eval = fcm.get_total_flops() - flops_forward_eval

    params_eval = sum(param.numel() for param in net_small.parameters())
    print(f"flops_forward={flops_forward_eval/1e9:.2f}G, flops_back={flops_backward_eval/1e9:.2f}G, params={params_eval/1e6:.2f} M")