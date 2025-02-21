import math
import collections
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

def freeze_module_params(m):
    if m is not None:
        for p in m.parameters():
            p.requires_grad = False

class MaskedConv1d(nn.Conv1d):
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int=1, dilation: int=1, groups: int=1,
                 bias: bool=True):
        padding = dilation * (kernel_size - 1) // 2
        super().__init__(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation,
                                           groups=groups, bias=bias, padding=padding)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        return super().forward(x.transpose(1, 2)).transpose(1, 2)

class ResnetBlock1d(nn.Module):
    def __init__(self, encoder_dim, hidden_size, kernel_size:int, activate='tanh'):
        super(ResnetBlock1d, self).__init__()
        if activate=='tanh':
            self.activate = nn.Tanh()
        elif activate=='relu':
            self.activate = nn.ReLU()
        else:
            self.activate = None
        self.cnn1 = MaskedConv1d(hidden_size, hidden_size, kernel_size)
        self.cnn2 = MaskedConv1d(hidden_size, hidden_size, kernel_size)

    def forward(self, x, input_mask=None):
        if input_mask is not None:
            x = x * input_mask
        identity = x
        x = self.cnn1(x)
        if self.activate is not None:
            x = self.activate(x)
        x = self.cnn2(x)
        x = identity + x
        if self.activate is not None:
            x = self.activate(x)
        return x

class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, linear=False):
        super().__init__()
        self.linear = linear
        if self.linear:
            self.layer = nn.Linear(in_dim, out_dim)

    def forward(self, x):
        if self.linear:
            x = F.relu(self.layer(x))
        x = torch.max(x, dim=1)[0]
        return x

class MLPMuformerDecoder(nn.Module):
    def __init__(self, encoder_dim, hidden_size, dropout=0.1):
        super(MLPMuformerDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.fc_pair = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
    # x1, x2: (B, T, C)
    def forward(self, x1, x2, **unused):
        x1 = self.fc_reduce(x1[:, 0, :])
        x2 = self.fc_reduce(x2[:, 0, :])
        x = torch.cat([x1, x2], dim=-1)
        return self.fc_pair(x)

class RawMuformerDecoder(nn.Module):
    def __init__(self, encoder_dim, hidden_size, dropout=0.0):
        super(RawMuformerDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.fc_attn = nn.Linear(self.hidden_size, 1)
        self.fc_pair = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
        self.fc_pair2 = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, 1)
        )
    # x: (B, T, C) --> (B, C)
    def weight_repr(self, x):
        weights = self.fc_attn(x)
        x_weighted = torch.bmm(x.transpose(1, 2), weights)
        x_weighted = x_weighted.squeeze(2)
        return x_weighted
    
    # x1, x2: (B, T, C)
    def forward(self, x1, x2, **unused):
        x1 = self.fc_reduce(x1)
        x2 = self.fc_reduce(x2)
        x1_weighted = self.weight_repr(x1)
        x2_weighted = self.weight_repr(x2)
        x1_sent = (x1[:, 0, :] + x1[:, -1, :]) / 2
        x2_sent = (x2[:, 0, :] + x2[:, -1, :]) / 2
        score_weighted = self.fc_pair(torch.cat([x1_weighted, x2_weighted], dim=-1))
        score_sent = self.fc_pair2(torch.cat([x1_sent, x2_sent], dim=-1))
        return score_weighted + score_sent

class ResnetMuformerDecoder(nn.Module):
    def __init__(self, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1):
        super(ResnetMuformerDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.resnet = nn.ModuleList([ResnetBlock1d(self.hidden_size, self.hidden_size, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(linear=True,in_dim=self.hidden_size,out_dim=self.hidden_size)
        self.output = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )
    
    def resnet_func(self, x, input_mask):
        for index, layer in enumerate(self.resnet):
            if index == 0:
                x = layer(x, input_mask)
            else:
                x = layer(x)
        return x
        
    # x1, x2: (B, T, C)
    def forward(self, x1, x2, **unused):
        x1 = self.reduce(x1)
        x1 = self.resnet_func(x1, input_mask=unused['x1_mask'][:,:,None].repeat(1, 1, self.hidden_size))
        x1 = self.pooling(x1 * unused['x1_mask'][:,:,None].repeat(1, 1, self.hidden_size))
        
        x2 = self.reduce(x2)
        x2 = self.resnet_func(x2, input_mask=unused['x2_mask'][:,:,None].repeat(1, 1, self.hidden_size))
        x2 = self.pooling(x2 * unused['x2_mask'][:,:,None].repeat(1, 1, self.hidden_size))
        
        out = self.output(torch.concat([x1, x2], dim=-1))
        return out

class SiameseMuformerDecoder(nn.Module):
    def __init__(self, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1):
        super(SiameseMuformerDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.motif_net = nn.ModuleList([ResnetBlock1d(self.hidden_size, self.hidden_size, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(linear=True,in_dim=self.hidden_size,out_dim=self.hidden_size)
        self.dist_motifs = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.dist_semantics = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

    # x1, x2: (B, T, C)
    def motif_feat(self, x, input_mask):
        for index, layer in enumerate(self.motif_net):
            if index == 0:
                x = layer(x, input_mask)
            else:
                x = layer(x)
        return x
        
    # x1, x2: (B, T, C)
    def distinguisher(self, x1, x2, input_mask1, input_mask2):
        x1 = self.fc_reduce(x1)
        x2 = self.fc_reduce(x2)

        x1_semantics = x1[:, 0, :]
        x2_semantics = x2[:, 0, :]

        x1_motifs = self.motif_feat(x1, input_mask=input_mask1)
        x2_motifs = self.motif_feat(x2, input_mask=input_mask2)

        x1_motifs = self.pooling(x1_motifs * input_mask1)
        x2_motifs = self.pooling(x2_motifs * input_mask2)
        
        delta_semantics = self.dist_semantics(torch.cat([x1_semantics, x2_semantics], dim=-1))
        delta_motifs = self.dist_motifs(torch.cat([x1_motifs, x2_motifs], dim=-1))

        return delta_semantics + delta_motifs

        # x1, x2: (B, T, C)
    def forward(self, x1, x2, **unused):
        if 'x1_mask' in unused:
            input_mask1 = unused['x1_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask1 = None
        if 'x2_mask' in unused:
            input_mask2 = unused['x2_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask2 = None

        return self.distinguisher(x1, x2, input_mask1, input_mask2)

class MonoMuformerDecoder(nn.Module):
    def __init__(self, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1):
        super(MonoMuformerDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.motif_net = nn.ModuleList([ResnetBlock1d(self.hidden_size, self.hidden_size, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(linear=True,in_dim=self.hidden_size,out_dim=self.hidden_size)
        self.dist_motifs = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.dist_semantics = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

    # x1, x2: (B, T, C)
    def motif_feat(self, x, input_mask=None):
        for index, layer in enumerate(self.motif_net):
            if index == 0:
                x = layer(x, input_mask)
            else:
                x = layer(x)
        return x
        
    # x1, x2: (B, T, C)
    def distinguisher(self, x1, x2, input_mask1, input_mask2):
        x1 = self.fc_reduce(x1)

        x1_semantics = x1[:, 0, :]

        x1_motifs = self.motif_feat(x1, input_mask=input_mask1)

        x1_motifs = self.pooling(x1_motifs * input_mask1)
        
        delta_semantics = self.dist_semantics(x1_semantics)
        delta_motifs = self.dist_motifs(x1_motifs)

        return delta_semantics + delta_motifs

        # x1, x2: (B, T, C)
    def forward(self, x1, x2, **unused):
        if 'x1_mask' in unused:
            input_mask1 = unused['x1_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask1 = None
        if 'x2_mask' in unused:
            input_mask2 = unused['x2_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask2 = None

        return self.distinguisher(x1, x2, input_mask1, input_mask2)
    
class Muformer(nn.Module):
    def __init__(self, 
        encoder=None, 
        encoder_dim=768, 
        hidden_size=256, 
        num_heads=16, 
        freeze_lm=False,
        biased_layers=None,
        dropout=0.1,
        decoder_name='raw',
        joint_mode_enabled=False,
        joint_method='weighted_sum'):
        super(Muformer, self).__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.biased_layers = biased_layers
        self.msa_feat_used = (biased_layers is not None)
        self.joint_mode_enabled = joint_mode_enabled
        self.joint_method = joint_method
        
        print(f'[ INFO ] MSA feat: \t [{self.msa_feat_used}]')
        print(f'[ INFO ] Hidden size: \t [{self.hidden_size}]')
        print(f'[ INFO ] Dropout: \t [{self.dropout}]')
        if freeze_lm:
            print('[ INFO ] Freezing LM encoder')
            freeze_module_params(self.encoder) ### Freeze the pre-trained LM model
        
        self.attn_map = nn.Sequential(
            nn.Linear(1, 1, bias=False),
            nn.GELU(),
            nn.Linear(1, num_heads, bias=False),
            # nn.Sigmoid()
        )
        self.decoder_dict = {
            'mlp': MLPMuformerDecoder,
            'raw': RawMuformerDecoder,
            'resnet': ResnetMuformerDecoder,
            'siamese': SiameseMuformerDecoder,
            'mono': MonoMuformerDecoder,
        }
        print(f'[ INFO ] Decoder: \t [{decoder_name}]')
        self.decoder = self.decoder_dict[decoder_name](encoder_dim, hidden_size, dropout=dropout)
        print(f'[ INFO ] Joint mode: \t [{self.joint_mode_enabled}]')
        self.weight_sum = nn.Linear(2, 1)

        self.mono = 'mono' in decoder_name
        
    def diff_prob(self, prob_lm1, prob_lm2, x1_indices, x2_indices):
        logits_lm1 = torch.log_softmax(prob_lm1, dim=-1)
        probs1 = torch.gather(logits_lm1, -1, x1_indices.unsqueeze(-1)).squeeze(-1)

        if self.mono:
            return probs1[:, 1:-1].mean(-1, keepdim=True)
        else:
            logits_lm2 = torch.log_softmax(prob_lm2, dim=-1)
            probs2 = torch.gather(logits_lm2, -1, x2_indices.unsqueeze(-1)).squeeze(-1)
            return (probs1[:, 1:-1].mean(-1, keepdim=True) - probs2[:, 1:-1].mean(-1, keepdim=True)) 
    
    def forward(self, x1, x2, msa_feat1, msa_feat2, **unused):
        
        if self.msa_feat_used:
            L1 = msa_feat1.size(-1)
            attn_bias1 = self.attn_map(msa_feat1.unsqueeze(-1)) # attn_bias1: (N, L1, L1, num_heads)
            attn_bias1 = attn_bias1.permute(0, 3, 1, 2).reshape(-1, L1, L1) # attn_bias1: (N * num_heads, L1, L1)
            
            if self.mono:
                attn_bias2 = None
            else:
                L2 = msa_feat2.size(-1)
                attn_bias2 = self.attn_map(msa_feat2.unsqueeze(-1)) # attn_bias2: (N, L2, L2, num_heads)
                attn_bias2 = attn_bias2.permute(0, 3, 1, 2).reshape(-1, L2, L2) # attn_bias2: (N * num_heads, L2, L2)
        else:
            attn_bias1 = None
            attn_bias2 = None
        
        x1_indices, x2_indices = x1, x2

        x1, _ = self.encoder(x1, attn_bias=attn_bias1, biased_layers=self.biased_layers)
        x1 = x1.permute(1, 0, 2)

        if self.mono:
            x2 = None
        else:
            x2, _ = self.encoder(x2, attn_bias=attn_bias2, biased_layers=self.biased_layers)
            x2 = x2.permute(1, 0, 2)
        output = self.decoder(x1, x2, **unused)
        
        if self.joint_mode_enabled:
            lm_prob1 = self.encoder.prob(x1)
            if self.mono:
                lm_prob2 = None
            else:
                lm_prob2 = self.encoder.prob(x2)
            diff = self.diff_prob(lm_prob1, lm_prob2, x1_indices, x2_indices)
            
            if self.joint_method == 'weighted_sum':
                output = self.weight_sum(torch.concat([output, diff],dim=-1))
            elif self.joint_method == 'average':
                output = (diff + output) / 2
            elif self.joint_method == 'add':
                output = diff + output
            elif self.joint_method == 'pure_diff':
                output = diff
            elif self.joint_method == 'multiply':
                output = diff * output
        return output
