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

def get_activation(activate):
    if activate=='tanh':
        return nn.Tanh()
    elif activate=='relu':
        return nn.ReLU()
    elif activate=='gelu':
        return nn.GELU()
    elif activate=='sigmoid':
        return nn.Sigmoid()
    elif activate=='softmax':
        return nn.Softmax()
    else:
        return None


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

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
        self.activate = get_activation(activate)
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

class RawMuformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_size, dropout=0.0, **unused):
        super(RawMuformerDecoder, self).__init__()
        self.vocab_size = vocab_size
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
        return score_sent, score_weighted

class SiameseMuformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1, **unused):
        super(SiameseMuformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.motif_net = nn.ModuleList([ResnetBlock1d(self.hidden_size, self.hidden_size, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(linear=True,in_dim=self.hidden_size,out_dim=self.hidden_size)
        self.dist_motifs = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            get_activation(activate),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.dist_semantics = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            get_activation(activate),
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
        
        s_sem = self.dist_semantics(torch.cat([x1_semantics, x2_semantics], dim=-1))
        s_motif = self.dist_motifs(torch.cat([x1_motifs, x2_motifs], dim=-1))

        return s_sem, s_motif

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

class EncoderOut():
    def __init__(self, encoder_out, encoder_padding_mask=None):
        self.encoder_out = encoder_out
        self.encoder_padding_mask = encoder_padding_mask

class Seq2SeqMuformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1, pretrained_decoder=None):
        super(Seq2SeqMuformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.pretrained_decoder = pretrained_decoder

        self.motif_net = nn.ModuleList([ResnetBlock1d(self.encoder_dim, self.encoder_dim, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(linear=True,in_dim=self.encoder_dim,out_dim=self.encoder_dim)

        self.dist_motifs = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_dim, self.encoder_dim),
            get_activation(activate),
            nn.Linear(self.encoder_dim, 1)
        )

        self.embedding = nn.Embedding(self.vocab_size, self.encoder_dim)
        self.pos_encoder = PositionalEncoding(self.encoder_dim, dropout)
        self.decoder = pretrained_decoder

        self.fc_attn = nn.Linear(self.encoder_dim, 1)
        self.fc_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.encoder_dim, self.encoder_dim),
            get_activation(activate),
            nn.Linear(self.encoder_dim, 1)
        )

    # x1, x2: (B, T, C)
    def motif_feat(self, x):
        for index, layer in enumerate(self.motif_net):
            if index == 0:
                x = layer(x)
            else:
                x = layer(x)
        return x

        # x1, x2: (B, T, C)
    def forward(self, x1, x2, x1_rawseq, x2_rawseq, **unused):
        
        # x2_rawseq = x2_rawseq.transpose(0, 1)
        x1 = x1.transpose(0, 1) # (B, T, C) --> (T, B, C)

        encoder_out = EncoderOut([x1])

        _, extra = self.decoder(x2_rawseq, encoder_out)

        x = extra['inner_states'][-1].permute(1, 0, 2)

        # x = x.transpose(0, 1)

        x_motifs = self.motif_feat(x)
        x_motifs = self.pooling(x_motifs)
        s_motif = self.dist_motifs(x_motifs)

        x = self.fc_out((x[:, 0, :] + x[:, -1, :]) / 2)

        return x, s_motif
        

class MonoMuformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1, **unused):
        super(MonoMuformerDecoder, self).__init__()
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.motif_net = nn.ModuleList([ResnetBlock1d(self.hidden_size, self.hidden_size, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(linear=True,in_dim=self.hidden_size,out_dim=self.hidden_size)

        # print('Debugging: ' + str(kernel_size) + ' - ' + str(layers))

        self.dist_motifs = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            get_activation(activate),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.dist_semantics = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            get_activation(activate),
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
    def distinguisher(self, x1, x2, input_mask1, input_mask2, **unused):
        x1 = self.fc_reduce(x1)

        x1_semantics = (x1[:, 0, :] + x1[:, -1, :]) / 2 # for gpt

        x1_motifs = self.motif_feat(x1, input_mask=input_mask1)

        x1_motifs = self.pooling(x1_motifs * input_mask1)
        
        s_sem = self.dist_semantics(x1_semantics)
        s_motif = self.dist_motifs(x1_motifs)
        
        return s_sem, s_motif

        # x1, x2: (B, T, C)
    def forward(self, x1, x2, x1_rawseq, x2_rawseq, **unused):
        if 'x1_mask' in unused:
            input_mask1 = unused['x1_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask1 = None
        if 'x2_mask' in unused:
            input_mask2 = unused['x2_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask2 = None

        return self.distinguisher(x1, x2, input_mask1, input_mask2, **unused)

class MonoXMuformerDecoder(nn.Module):
    def __init__(self, vocab_size, encoder_dim, hidden_size, activate='tanh', dropout=0.1, kernel_size=3, layers=1):
        super(MonoXMuformerDecoder, self).__init__()
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.dropout = dropout
        self.fc_reduce = nn.Linear(self.encoder_dim, self.hidden_size)
        self.rawseq_fc_reduce = nn.Linear(self.vocab_size, self.hidden_size)
        self.motif_net = nn.ModuleList([ResnetBlock1d(self.hidden_size, self.hidden_size, kernel_size, activate=activate) for _ in range(layers)])
        self.pooling = LengthMaxPool1D(linear=True,in_dim=self.hidden_size,out_dim=self.hidden_size)

        self.rawseq_motif_net = ResnetBlock1d(self.vocab_size, self.hidden_size, kernel_size, activate=activate)
        self.rawseq_pooling = LengthMaxPool1D(linear=True,in_dim=self.hidden_size,out_dim=self.hidden_size)
        self.rawseq_out = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.dist_motifs = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            get_activation(activate),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, 1)
        )

        self.dist_semantics = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_size, self.hidden_size),
            get_activation(activate),
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
        x = self.pooling(x * input_mask)
        return x
    
    # x1, x2: (B, T, vocab_size)
    def rawseq_motif_feat(self, x, input_mask):
        x = self.rawseq_motif_net(x, input_mask)
        x = self.rawseq_pooling(x * input_mask)
        return x
        
    # x1, x2: (B, T, C)
    def distinguisher(self, x1, input_mask1, x1_rawseq, **unused):
        x1 = self.fc_reduce(x1)

        x1_rawseq = F.one_hot(x1_rawseq, num_classes=self.vocab_size).float()
        x1_rawseq = self.rawseq_fc_reduce(x1_rawseq)

        x1_semantics = x1[:, 0, :]

        x1_motifs = self.motif_feat(x1, input_mask=input_mask1)
        x1_rawseq_motifs = self.rawseq_motif_feat(x1_rawseq, input_mask1)

        s_rawseq_motif = self.rawseq_out(x1_rawseq_motifs)
        
        s_sem = self.dist_semantics(x1_semantics)
        s_motif = self.dist_motifs(x1_motifs)
        
        return s_sem, s_motif + s_rawseq_motif

    # x1, x2: (B, T, C)
    def forward(self, x1, x2, x1_rawseq, x2_rawseq, **unused):
        if 'x1_mask' in unused:
            input_mask1 = unused['x1_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask1 = None
        if 'x2_mask' in unused:
            input_mask2 = unused['x2_mask'][:,:,None].repeat(1, 1, self.hidden_size)
        else:
            input_mask2 = None

        return self.distinguisher(x1, input_mask1, x1_rawseq, **unused)

class Muformer(nn.Module):
    def __init__(self, 
        encoder=None, 
        vocab_size=-1, 
        encoder_dim=768, 
        hidden_size=256, 
        num_heads=16, 
        freeze_lm=False,
        biased_layers=None,
        dropout=0.1,
        encoder_name='pmlm',
        decoder_name='mono',
        joint_mode_enabled=False,
        joint_method='weighted_sum',
        activate='tanh',
        kernel_size=3,
        conv_layers=1,
        pretrained_decoder=None):

        super(Muformer, self).__init__()
        self.encoder = encoder
        self.vocab_size = vocab_size
        self.encoder_dim = encoder_dim
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.biased_layers = biased_layers
        self.msa_feat_used = (biased_layers is not None)
        self.joint_mode_enabled = joint_mode_enabled
        self.joint_method = joint_method
        self.encoder_name = encoder_name
        self.kernel_size = kernel_size
        self.pretrained_decoder = pretrained_decoder
        
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
        )
        self.decoder_dict = {
            'raw': RawMuformerDecoder,
            'siamese': SiameseMuformerDecoder,
            'mono': MonoMuformerDecoder,
            'monox': MonoXMuformerDecoder,
            'seq2seq': Seq2SeqMuformerDecoder,
        }
        print(f'[ INFO ] Decoder: \t [{decoder_name}]')
        self.decoder = self.decoder_dict[decoder_name](vocab_size, encoder_dim, hidden_size, dropout=dropout, activate=activate, kernel_size=kernel_size, layers=conv_layers, pretrained_decoder=pretrained_decoder)
        print(f'[ INFO ] Vocab size: \t [{vocab_size}]')
        print(f'[ INFO ] Joint mode: \t [{self.joint_mode_enabled}]')
        self.weight_sum = nn.Linear(3, 1)

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
    
    def forward(self, x1, x2, msa_feat1=None, msa_feat2=None, **unused):

        x1_rawseq = x1
        x2_rawseq = x2
        # print("msa feature used or not: ", self.msa_feat_used)
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

        lm_prob1, lm_prob2 = None, None
        
        # print("encoder name: ", self.encoder_name)
        # print("mono: ", self.mono)
        if self.encoder_name == 'pmlm' or 'pmlm' in self.encoder_name:
            x1, _ = self.encoder(x1, attn_bias=attn_bias1, biased_layers=self.biased_layers)
            x1 = x1.permute(1, 0, 2)
            analysis_embedding = x1[0,0,:]
            # print("analysis embedding shape: ", analysis_embedding.shape)

            if self.mono:
                x2 = None
            else:
                x2, _ = self.encoder(x2, attn_bias=attn_bias2, biased_layers=self.biased_layers)
                x2 = x2.permute(1, 0, 2)
        elif self.encoder_name == 'gpt':
            lm_prob1, extra1 = self.encoder(x1)
            x1 = extra1['inner_states'][-1].permute(1, 0, 2)

            if self.mono:
                x2 = None
            else:
                lm_prob2, extra2 = self.encoder(x2)
                x2 = extra2['inner_states'][-1].permute(1, 0, 2)
        elif self.encoder_name == 'esm' or self.encoder_name == 'esm2':
            x1_all = self.encoder(x1,repr_layers=[self.encoder.args.layers])
            x1 = x1_all['representations'][self.encoder.args.layers]
            if self.mono:
                x2 = None
            else:
                x2_all = self.encoder(x2,repr_layers=[self.encoder.args.layers])
                x2 = x2_all['representations'][self.encoder.args.layers]
        elif self.encoder_name == 'esm1':
            x1_all = self.encoder(x1,repr_layers=[self.encoder.num_layers])
            x1 = x1_all['representations'][self.encoder.num_layers]
            if self.mono:
                x2 = None
            else:
                x2_all = self.encoder(x2,repr_layers=[self.encoder.num_layers])
                x2 = x2_all['representations'][self.encoder.num_layers]
        
        s_sem, s_motif = self.decoder(x1, x2, x1_rawseq, x2_rawseq, **unused)

        # print(self.joint_mode_enabled)
        # print(self.joint_method)
        if self.joint_mode_enabled:
            if self.encoder_name == 'pmlm' or 'pmlm' in self.encoder_name:
                lm_prob1 = self.encoder.prob(x1)
                if self.mono:
                    lm_prob2 = None
                else:
                    lm_prob2 = self.encoder.prob(x2)
            elif self.encoder_name == 'gpt':
                pass
            elif self.encoder_name in ['esm', 'esm1', 'esm2']:
                lm_prob1 = x1_all["logits"]
                if self.mono:
                    lm_prob2 = None
                else:
                    lm_prob2 = x2_all["logits"]
            s_prob = self.diff_prob(lm_prob1, lm_prob2, x1_indices, x2_indices)

            if self.joint_method == 'weighted_sum':
                # print("s_sem: ", s_sem.shape)
                # print("s_motif: ", s_motif.shape)
                # print("s_prob: ", s_prob.shape)
                output = self.weight_sum(torch.concat([s_sem, s_motif, s_prob],dim=-1))
            elif self.joint_method == 'average':
                # output = (s_sem + s_motif + s_prob) / 3
                output = (s_sem + s_motif + s_prob) / 2
            elif self.joint_method == 'multiply':
                output = s_sem * s_motif * s_prob
            elif self.joint_method == 'no_motif':
                output = (s_sem + s_prob) / 2
            elif self.joint_method == 'no_prob':
                output = (s_sem + s_motif) / 2
            elif self.joint_method == 'no_sem':
                output = (s_motif + s_prob) / 2
            elif self.joint_method == 'only_motif':
                output = s_motif
            elif self.joint_method == 'only_prob':
                output = s_prob
            elif self.joint_method == 'only_sem':
                output = s_sem
            else:
                output = s_sem + s_motif + s_prob
        
            return [output, analysis_embedding]
 
