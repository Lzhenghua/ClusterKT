import torch
from torch import nn
from torch.nn.init import xavier_uniform_,kaiming_normal_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from torch.nn import Parameter
import seaborn as sb

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class ClusterKT(nn.Module):
    def __init__(self, n_question, n_pid, d_model,
                 n_blocks, kq_same, dropout, model_type,
                 cluster_size, final_fc_dim, n_heads,
                 d_ff, n_st, n_et, separate_qa=False,attn_=True):
        super().__init__()
        self.n_question = n_question
        self.dropout = dropout
        self.kq_same = kq_same  # Whether k and q come from the same learning sequence.
        self.n_pid = n_pid  # Represent the answers as two vectors or a embedding matrix.
        self.model_type = model_type
        self.separate_qa = separate_qa
        self.n_st = n_st
        self.n_et = n_et
        self.d_model = d_model
        self.attn_ = attn_

        # Whether to represent the discrepancy in exercises and answers.
        if self.n_pid > 0:
            self.difficult_param = nn.Embedding(self.n_pid + 1, 1)
            self.q_embed_diff = nn.Embedding(self.n_question + 1, self.d_model)
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, self.d_model)

        self.q_embed = nn.Embedding(self.n_question + 1, self.d_model)
        self.qa_embed = nn.Embedding(2 * self.n_question + 2, self.d_model)
        self.interval_embed = nn.Embedding(self.interval + 2, self.d_model, self.interval + 1)
        self.time_embed = nn.Embedding(self.time + 2, self.d_model, padding_idx=self.time + 1)

        self.decoder_map = nn.Linear(self.d_model * 3, self.d_model)
        self.encoder_map = nn.Linear(self.d_model * 3, self.d_model)
        self.lg_unit = learning_gain_unit(dropout,self.d_model)
        self.rnn_attn_map = nn.Linear(self.d_model * 2, self.d_model)
        self.trans = Architecture(n_blocks=n_blocks, n_heads=n_heads,
                                  dropout=dropout, d_model=self.d_model,
                                  d_ff=d_ff, kq_same=self.kq_same, model_type=self.model_type)

        self.n_center = cluster_size
        self.concept_center = Parameter(torch.Tensor(self.n_center, d_model), requires_grad=True)
        self.state_center = Parameter(torch.Tensor(self.n_center, d_model), requires_grad=True)
        kaiming_normal_(self.concept_center)
        kaiming_normal_(self.state_center)

        self.add_gate = nn.Linear(d_model, d_model)
        self.erase_gate = nn.Linear(d_model, d_model)
        self.forget_para_map = nn.Linear(2 * d_model, 1)
        self.sig = nn.Sigmoid()
        
        self.pre_attn = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=n_heads, dropout=dropout, batch_first=True)
        self.pre_attn_ = nn.MultiheadAttention(embed_dim=self.d_model, num_heads=n_heads, dropout=dropout, batch_first=True)

        self.mlp = nn.Sequential(
            nn.Linear(self.d_model + self.d_model, final_fc_dim),
            nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(final_fc_dim, 256),
            nn.ReLU(), nn.Dropout(self.dropout), nn.Linear(256, 1))
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def forward(self, q_data, qa_data, pid_data, dotime, lagtime):
        q_embed_data = self.q_embed(q_data)

        # Embed students' answers.
        if self.separate_qa:
            qa_embed_data = self.qa_embed(qa_data)
        else:
            qa_data = (qa_data - q_data) // self.n_question
            qa_embed_data = self.qa_embed(qa_data) + q_embed_data

        # Embed the discrepancy over exercises and answers via the Rasch model.
        if self.n_pid > 0:
            q_embed_diff_data = self.q_embed_diff(q_data)
            pid_embed_data = self.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            qa_embed_diff_data = self.qa_embed_diff(qa_data)
            if self.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff_data
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (qa_embed_diff_data + q_embed_diff_data)

        # Embed the time-related information:spent time, elapsed time. And concatenate them with answers and exercises.
        st = self.time_embed(dotime)
        et = self.interval_embed(lagtime)
        q_embed_data = self.encoder_map(torch.cat((q_embed_data, st, et),dim=-1))
        qa_embed_data = self.decoder_map(torch.cat((qa_embed_data, st, et),dim=-1))
        batch_size = q_embed_data.size(0)
        seq_len = q_embed_data.size(1)
        b_concept_center = self.concept_center.unsqueeze(0).repeat(batch_size, 1, 1)
        b_state_center = self.state_center.unsqueeze(0).repeat(batch_size, 1, 1)

        q_ori = q_embed_data.clone().detach().permute(1,0,2) # 100,24,256
        q_ori = torch.unsqueeze(q_ori,dim=-1) # 100,24,256,1
        qa_ori = qa_embed_data.clone().detach().permute(1, 0, 2)
        qa_ori = torch.unsqueeze(qa_ori, dim=-1)
        h_add = torch.tanh(self.add_gate(qa_embed_data)) # 100, 24, 256, 1
        h_minus = torch.sigmoid(self.erase_gate(qa_embed_data))
        h_list = []
        cluster_loss_list = []
        sum_center = b_concept_center.clone().detach()
        all_state_sim = []

        # Exercise clusters and Performance clusters
        for i in range(seq_len):
            sim = F.cosine_similarity(q_ori[i,:,:], b_concept_center.permute(0,2,1), dim=1)
            idx = torch.sort(sim, dim=-1, descending=True).indices[:, 0]
            b_loss = torch.sort(sim, dim=-1, descending=True).values[:, 0]
            cluster_loss_list.append(torch.sum(1-b_loss))
            idx_center = sum_center[torch.arange(batch_size),idx,:]
            tempt = q_ori[i,:,:].squeeze(-1) + idx_center
            cluster_norm = torch.norm(tempt,p=1,dim=-1).unsqueeze(dim=-1)
            center = torch.div(tempt,cluster_norm)
            b_concept_center[torch.arange(batch_size), idx, :] = center
            sum_center[torch.arange(batch_size), idx, :] = tempt
            
            batch_idx_state = b_state_center[torch.arange(batch_size),idx,:]
            h_list.append(batch_idx_state)
            batch_idx_state_ = torch.unsqueeze(batch_idx_state,dim=-1)
            state_sim = F.cosine_similarity(batch_idx_state_, b_state_center.permute(0,2,1), dim=1)
            all_state_sim.append(state_sim)
            batch_idx_state = batch_idx_state * (1 - h_minus[:,i,:]) + h_add[:,i,:] * qa_ori[i, :, :].squeeze(-1)
            b_state_center[torch.arange(batch_size),idx,:] = batch_idx_state

        cluster_state = torch.stack(h_list, dim=1)  # 24, 100, 256
        sim_num = cluster_state.clone().detach().permute(1,0,2).unsqueeze(-1)
        state_sim_list = []
        for i in range(seq_len):
            sim = F.cosine_similarity(sim_num[i, :, :], cluster_state.permute(0,2,1), dim=1)
            state_sim_list.append(sim)
        all_sim = torch.stack(state_sim_list, dim=1) # 24,100,100
        forget_difficulty = self.sig(self.forget_map(torch.cat((cluster_state, q_embed_diff_data),dim=-1))) # 24,100,256

        learning_states = self.lg_unit(q_embed_data, cluster_state, all_sim)
        forgetting_states = self.trans(q_embed_data, learning_states, all_sim, forget_difficulty)

        # Predict students' performance on current question via the mlp and attention layer.
        mask = ut_mask(learning_states.size(1))
        trans_output, _ = self.pre_attn(learning_states, forgetting_states, cluster_state, attn_mask=mask)
        knowledge_states, _ = self.pre_attn_(learning_states, trans_output, cluster_state, attn_mask=mask)
        concat_q = torch.cat([trans_output, q_embed_data], dim=-1)
        output = self.mlp(concat_q)
        x = torch.sigmoid(output)
        return x.squeeze(-1), cluster_loss_list

def ut_mask(seq_len):
    return torch.triu(torch.ones(seq_len,seq_len),diagonal=1).to(dtype=torch.bool)
    
class learning_gain_unit(nn.Module):
    def __init__(self, dropout, d_model):
        super(learning_gain_unit, self).__init__()
        self.d_model = d_model
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        self.exp = nn.Parameter(nn.init.xavier_uniform_(torch.empty(1, self.d_model)), requires_grad=True)
        self.linear_gate = nn.Linear(2 * self.d_model, self.d_model)
        self.linear_k = nn.Linear(2 * self.d_model, self.d_model)
        self.linear_re = nn.Linear(2 * self.d_model, self.d_model)

    def forward(self, q_embed_data, cluster_state, all_sim):
        B,L,d = cluster_state.shape
        padding = torch.zeros(B, 1, d)
        a_emb = torch.cat((padding, cluster_state[:, :-1, :]), 1)
        qa = a_emb.split(1, dim=1)
        q_embed_data = torch.cat((padding, q_embed_data[:, :-1, :]), 1)
        q = q_embed_data.split(1, dim=1)
        exp = self.exp.repeat(q_embed_data.size(0), 1).cuda()
        learning_gain_list = list()
        learning_gain_list.append(exp.unsqueeze(1))
        seqlen = 99
        all_sim_slot = all_sim.split(1, dim=1)
        for i in range(1, seqlen+1):
            a_1 = torch.squeeze(qa[i], 1)
            an_input = torch.squeeze(q[i], 1)
            x = torch.cat((a_1, an_input),dim=-1)
            gate = self.sigmoid(self.linear_gate(x))
            k_q = self.tanh(self.linear_k(x))
            gain = gate * k_q
            ins = torch.cat((a_1, an_input), -1)
            reset = self.sigmoid(self.linear_re(ins))
            exp = reset * exp + (1 - reset) * gain * all_sim_slot
            lg = torch.unsqueeze(exp, dim=1)
            learning_gain_list.append(lg)
        learning_gain = torch.cat(learning_gain_list, axis=1)
        return learning_gain

class Architecture(nn.Module):
    def __init__(self, n_blocks, d_model, d_ff, n_heads, dropout, kq_same, model_type):
        super(Architecture, self).__init__()
        '''
        This is a transformer layer with n_blocks encoders and decoders.
        '''
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'ClusterKT'}:
            self.blocks_1 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)    # Decoders
            ])
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks * 2)    # Encoders and the cross-attention
            ])

    def forward(self, q_embed_data, qa_embed_data, satate_sim, decay):
        y = qa_embed_data
        x = q_embed_data
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, satate_sim=satate_sim, decay=decay)
        flag_first = True
        for block in self.blocks_2:
            if flag_first: 
                x = block(mask=1, query=x, key=x,
                          values=x, satate_sim=satate_sim, decay=decay, apply_pos=False)
                flag_first = False
            else: 
                x = block(mask=0, query=x, key=x, values=y, satate_sim=satate_sim, decay=decay, apply_pos=True)
                flag_first = True
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        '''
        The encoders or decoders is composed of the multi-head self-attention, the Add & Norm, and the FFN.
        '''
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, satate_sim, decay, apply_pos=True):
        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0)
        if mask == 0:
            query2 = self.masked_attn_head(
                query, key, values, satate_sim, decay, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(
                query, key, values, satate_sim, decay, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)

        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, satate_sim, decay, mask, zero_pad):
        bs = q.size(0)

        # perform linear operation and split into h heads
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        gammas = self.gammas
        scores = forgetting_attention(q, k, v, self.d_k, mask, self.dropout, zero_pad, satate_sim, decay,gammas)

        # Concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)

def forgetting_attention(q, k, v, d_k, mask, dropout, zero_pad, satate_sim, difficulty):
    scores = (torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k))
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = (F.softmax(scores_, dim=-1))
        scores_ = (scores_ * mask.float())
        distcum_scores = torch.cumsum(scores_, dim=-1)  # bs, 4, sl, sl
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)  # bs, 4, sl, 1
        position_effect = torch.unsqueeze(satate_sim,1)
        dist_scores = torch.clamp((disttotal_scores-distcum_scores)*position_effect, min=0.)
        dist_scores = dist_scores.sqrt().detach() # torch.Size([24, 4, 100, 100])
        m = nn.Softplus()
        gamma = torch.unsqueeze(difficulty,1)
        gamma = -1. * m(gamma)

    # Now after do exp(gamma*distance) and then clamp to 1e-5 to 1e5
        total_effect = torch.clamp(torch.clamp((dist_scores*gamma).exp(), min=1e-5), max=1e5) # 24, 4, 100, 100
        scores = scores * total_effect
        scores.masked_fill_(mask == 0, -1e32)
        scores = F.softmax(scores, dim=-1)  # BS,4,seqlen,seqlen
        if zero_pad:
            pad_zero = torch.zeros(bs, head, 1, seqlen)
            scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
        scores = dropout(scores)
        
        return torch.matmul(scores, v)

class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]

class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]