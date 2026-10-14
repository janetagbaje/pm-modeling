import numpy as np

import torch
import torch.nn as nn
from einops import repeat
from fairscale.nn import checkpoint_wrapper


# The code to build the model is modified from:
# https://github.com/krasserm/perceiver-io



class Sequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self:
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                inputs = module(inputs)
        return inputs


def mlp(num_channels: int):
    return Sequential(
                        nn.LayerNorm(num_channels),
                        nn.Linear(num_channels, num_channels),
                        nn.GELU(),
                        nn.Linear(num_channels, num_channels),
                    )


def cross_attention_layer( num_q_channels: int, 
                           num_kv_channels: int, 
                           num_heads: int, 
                           dropout: float, 
                           activation_checkpoint: bool = False):
    
    layer = Sequential(
        Residual(CrossAttention(num_q_channels, num_kv_channels, num_heads, dropout), dropout),
        Residual(mlp(num_q_channels), dropout),
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_layer(num_channels: int, 
                         num_heads: int, 
                         dropout: float, 
                         activation_checkpoint: bool = False):
    
    layer = Sequential(
        Residual(SelfAttention(num_channels, num_heads, dropout), dropout), 
        Residual(mlp(num_channels), dropout)
    )
    return layer if not activation_checkpoint else checkpoint_wrapper(layer)


def self_attention_block(num_layers: int, 
                         num_channels: int, 
                         num_heads: int, 
                         dropout: float, 
                         activation_checkpoint: bool = False
                        ):
    
    layers = [self_attention_layer(
                            num_channels, 
                            num_heads, 
                            dropout, 
                            activation_checkpoint) for _ in range(num_layers)]
    
    return Sequential(*layers)


class Residual(nn.Module):
    def __init__(self, module: nn.Module, dropout: float):
        super().__init__()
        self.module = module
        self.dropout = nn.Dropout(p=dropout)
        self.dropout_p = dropout

    def forward(self, *args, **kwargs):
        x = self.module(*args, **kwargs)
        return self.dropout(x) + args[0]


class MultiHeadAttention(nn.Module):
    
    def __init__(self, num_q_channels: int, 
                 num_kv_channels: int, 
                 num_heads: int, 
                 dropout: float):
        
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=num_q_channels,
            num_heads=num_heads,
            kdim=num_kv_channels,
            vdim=num_kv_channels,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        return self.attention(x_q, x_kv, x_kv, key_padding_mask=pad_mask, attn_mask=attn_mask)[0]


class CrossAttention(nn.Module):
    
    def __init__(self, 
                 num_q_channels: int, 
                 num_kv_channels: int, 
                 num_heads: int, 
                 dropout: float):
        
        super().__init__()
        self.q_norm = nn.LayerNorm(num_q_channels)
        self.kv_norm = nn.LayerNorm(num_kv_channels)
        self.attention = MultiHeadAttention(
                                            num_q_channels=num_q_channels, 
                                            num_kv_channels=num_kv_channels, 
                                            num_heads=num_heads, 
                                            dropout=dropout
                                            )

    def forward(self, x_q, x_kv, pad_mask=None, attn_mask=None):
        x_q = self.q_norm(x_q)
        x_kv = self.kv_norm(x_kv)
        return self.attention(x_q, x_kv, pad_mask=pad_mask, attn_mask=attn_mask)


class SelfAttention(nn.Module):
    def __init__(self, num_channels: int, num_heads: int, dropout: float):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels)
        self.attention = MultiHeadAttention(
                                            num_q_channels=num_channels, 
                                            num_kv_channels=num_channels, 
                                            num_heads=num_heads, 
                                            dropout=dropout
                                            )

    def forward(self, x, pad_mask=None, attn_mask=None):
        x = self.norm(x)
        return self.attention(x, x, pad_mask=pad_mask, attn_mask=attn_mask)


class Encoder(nn.Module):
    
  
  
    def __init__(
        self,
        input_ch,
        preproc_ch,
        
         
        num_latents: int,
        num_latent_channels: int,
        num_layers: int = 3,
        num_cross_attention_heads: int = 4,
        num_self_attention_heads: int = 4,
        num_self_attention_layers_per_block: int = 6,
        dropout: float = 0.0,
        activation_checkpoint: bool = False,
    ):
       
        super().__init__()
        self.time_enc_sin = SineActivation(1,12)
        self.time_enc_cos = CosineActivation(1,12)
        self.land_enc = nn.Embedding(17, 12)
        self.num_layers = num_layers
        if preproc_ch:
            self.preproc = nn.Linear(input_ch+36, preproc_ch)
        else:
            self.preproc = None
            preproc_ch   = input_ch
            
        def create_layer():
            return Sequential(
                cross_attention_layer(
                    num_q_channels=num_latent_channels,
                    num_kv_channels=preproc_ch, 
                    num_heads=num_cross_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
                self_attention_block(
                    num_layers=num_self_attention_layers_per_block,
                    num_channels=num_latent_channels,
                    num_heads=num_self_attention_heads,
                    dropout=dropout,
                    activation_checkpoint=activation_checkpoint,
                ),
            )

        self.layer_1 = create_layer()

        if num_layers > 1:
            self.layer_n = create_layer()
        self.latent = nn.Parameter(torch.empty(num_latents, num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.latent.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, pad_mask=None):
        b, *_ = x.shape
        time_encoding = torch.concat([self.time_enc_cos(x[:,0,-1].unsqueeze(-1)),
                                      self.time_enc_sin(x[:,0,-1].unsqueeze(-1))],
                                      dim=-1).unsqueeze(1).repeat_interleave(x.shape[1],dim=1)
        land_encoding = self.land_enc(x[:,:,-2].int())
        x = torch.concat([x[:,:,:-2],land_encoding, time_encoding],dim=-1)
        if self.preproc:
            x = self.preproc(x)

        # repeat initial latent vector along batch dimension
        x_latent = repeat(self.latent, "... -> b ...", b=b)
         
        x_latent = self.layer_1(x_latent, x, pad_mask)
        for i in range(self.num_layers - 1):
            x_latent = self.layer_n(x_latent, x, pad_mask)

        return x_latent


class Decoder(nn.Module):
    def __init__(
        self,
        ff_channels: int,
        preproc_ch,
        num_latent_channels: int,
        latent_size,
        num_output_channels,
        num_cross_attention_heads: int = 4,
        dropout: float = 0.0,
        activation_checkpoint: bool = False,
    ):
        
        super().__init__()
        q_chan = ff_channels + num_latent_channels
        if preproc_ch:
            q_in = preproc_ch
        else:
            q_in = q_chan


        self.postproc = nn.Linear(q_in, num_output_channels)
        
        if preproc_ch:
            self.preproc = nn.Linear(q_chan, preproc_ch)
        else:
            self.preproc = None
        
        self.cross_attention = cross_attention_layer(
                                    num_q_channels=q_in,
                                    num_kv_channels=num_latent_channels,
                                    num_heads=num_cross_attention_heads,
                                    dropout=dropout,
                                    activation_checkpoint=activation_checkpoint,
                                )

        self.output = nn.Parameter(torch.empty(latent_size,num_latent_channels))
        self._init_parameters()

    def _init_parameters(self):
        with torch.no_grad():
            self.output.normal_(0.0, 0.02).clamp_(-2.0, 2.0)

    def forward(self, x, coords):
        b, *_ = x.shape
        output = repeat(self.output, "... -> b ...", b=b)
        output = torch.repeat_interleave(output, coords.shape[1], axis=1)
        
        output = torch.cat([coords,output], axis=-1) 
        
        if self.preproc:
            output = self.preproc(output)
            
        output = self.cross_attention(output, x)
        return self.postproc(output)

# time2vec (t2v) from https://github.com/ojus1/Time2Vec-PyTorch
def t2v(tau, f, out_features, w, b, w0, b0, arg=None):
    if arg:
        v1 = f(torch.matmul(tau, w) + b, arg)
    else:
        v1 = f(torch.matmul(tau, w) + b)
    v2 = torch.matmul(tau, w0) + b0
    return torch.cat([v1, v2], -1)

class SineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(SineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.sin

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)

class CosineActivation(nn.Module):
    def __init__(self, in_features, out_features):
        super(CosineActivation, self).__init__()
        self.out_features = out_features
        self.w0 = nn.parameter.Parameter(torch.randn(in_features, 1))
        self.b0 = nn.parameter.Parameter(torch.randn(1))
        self.w = nn.parameter.Parameter(torch.randn(in_features, out_features-1))
        self.b = nn.parameter.Parameter(torch.randn(out_features-1))
        self.f = torch.cos

    def forward(self, tau):
        return t2v(tau, self.f, self.out_features, self.w, self.b, self.w0, self.b0)
