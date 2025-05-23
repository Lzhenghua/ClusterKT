import torch
import torch.nn as nn

class KTModel(nn.Module):
  def __init__(self, n_skill, n_diff, n_hints, n_time_used, max_len, emb_dim):
    super().__init__()
    self.max_len = max_len
    self.emb_dim = emb_dim

    self.skill_emb = nn.Embedding(n_skill+2, emb_dim, padding_idx=n_skill+1)
    self.answer_emb = nn.Embedding(3, emb_dim, padding_idx=2)
    self.diff_emb = nn.Embedding(n_diff+2, emb_dim, padding_idx=n_diff+1)
    self.hints_emb = nn.Embedding(n_hints+2, emb_dim, padding_idx=n_hints+1)
    self.time_used_emb = nn.Embedding(n_time_used+2, emb_dim, padding_idx=n_time_used+1)
    self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # self.device = torch.device('cpu')

    self.W_q_1 = nn.Linear(emb_dim, emb_dim, bias=False)
    self.W_k_1 = nn.Linear(emb_dim, emb_dim, bias=False)
    self.W_v_1 = nn.Linear(emb_dim, emb_dim, bias=False)

    self.W_q_2 = nn.Linear(emb_dim, emb_dim, bias=False)
    self.W_k_2 = nn.Linear(emb_dim, emb_dim, bias=False)
    self.W_v_2 = nn.Linear(emb_dim, emb_dim, bias=False)

    self.W_q_3 = nn.Linear(emb_dim, emb_dim, bias=False)
    self.W_k_3 = nn.Linear(emb_dim, emb_dim, bias=False)
    self.W_v_3 = nn.Linear(emb_dim, emb_dim, bias=False)


    self.fc = nn.Linear(2*emb_dim, emb_dim)
    self.fc2 = nn.Linear(2*emb_dim, emb_dim)
    self.fusion1_1 = nn.Linear(emb_dim, emb_dim)
    self.fusion1_2 = nn.Linear(emb_dim, emb_dim)
    self.fusion2_1 = nn.Linear(emb_dim, emb_dim)
    self.fusion2_2 = nn.Linear(emb_dim, emb_dim)
    
  def forward(self, skill, answer, diff, hints, time_used):
    # input: (batch_size, seq_len, emb_dim)
    bs = skill.size(0)
    embed_skill = self.skill_emb(skill)
    embed_diff = self.diff_emb(diff)
    embed_a = self.answer_emb(answer)
    embed_hints = self.hints_emb(hints)
    embed_time_used = self.time_used_emb(time_used)

    x = torch.cat((embed_skill, embed_diff), 2)
    x = self.fc(x)
    assert x.size(2) == self.emb_dim

    h_t, c_t = (torch.zeros(bs, self.emb_dim).to(self.device),
                torch.zeros(bs, self.emb_dim).to(self.device))

    y_preds = []

    for t in range(self.max_len-1):
      x_t = x[:, t, :] # batch_size, emb_dim
      x_next = x[:, t+1, :] # batch_size, emb_dim

      # 3 user responses
      used_t = embed_time_used[:, t, :]
      hints_t = embed_hints[:, t, :]
      a_t = embed_a[:, t, :]

      x_t = self.fc2(torch.cat((x_t, h_t), 1))

      q_1 = self.W_q_1(x_t)
      k_1 = self.W_k_1(c_t)
      v_1 = self.W_v_1(used_t)
      score_1 = torch.sigmoid(q_1 * k_1 / torch.sqrt(torch.ones(bs, self.emb_dim).to(self.device) * self.emb_dim))
      out_1 = v_1 * score_1

      q_2 = self.W_q_2(x_t)
      k_2 = self.W_k_2(c_t)
      v_2 = self.W_v_2(hints_t)
      score_2 = torch.sigmoid(q_2 * k_2 / torch.sqrt(torch.ones(bs, self.emb_dim).to(self.device) * self.emb_dim))
      out_2 = v_2 * score_2

      # fusion gate 1
      out = out_1 + out_2
      out = out_2
      c_t = torch.tanh(self.fusion1_1(out)) * torch.sigmoid(self.fusion1_2(out)) + c_t

      q_3 = self.W_k_3(x_t)
      k_3 = self.W_q_3(h_t)
      v_3 = self.W_v_3(a_t)
      score_3 = torch.sigmoid(q_3 * k_3 / torch.sqrt(torch.ones(bs, self.emb_dim).to(self.device) * self.emb_dim))
      out_3 = v_3 * score_3

      h_t = torch.tanh(self.fusion2_1(out_3)) * torch.sigmoid(self.fusion2_2(out_3)) + c_t
      _y_pred = torch.sigmoid(torch.sum(x_next * h_t, dim=1))
      y_preds.append(_y_pred)

    y_preds.append(torch.zeros(bs).to(self.device))
    y_pred = torch.stack(y_preds, dim=1)
    return y_pred
