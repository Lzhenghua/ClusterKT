import torch.utils.data
import torch.nn.utils
import numpy as np
from torch.utils.data import Dataset

class ClusterKT_dataset(Dataset):
    def __init__(self, group, n_ques, n_concept,max_seq):
        self.samples = group
        self.n_ques = n_ques
        self.max_seq = max_seq
        self.data = []
        self.n_concept = n_concept

        for que, exe_cat, ans, timestamp, spendtime in self.samples:
            if len(que) >= self.max_seq:
                self.data.extend([(que[l:l + self.max_seq], ans[l:l + self.max_seq],
                                   exe_cat[l:l + self.max_seq],timestamp[l:l + self.max_seq],
                                   spendtime[l:l + self.max_seq]) for l in range(len(que)) if l % self.max_seq == 0])
            elif len(que) <= self.max_seq and len(que) > 50:
                self.data.append((que, ans, exe_cat,timestamp,spendtime))
            else:
                continue

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ques_ids, answered_correctly, concept_ids, timestamp, spendtime = self.data[idx]
        ques_ids = np.array(list(map(int, ques_ids)))
        answered_correctly = np.array(list(map(int, answered_correctly)))
        concept_ids = np.array(list(map(int, concept_ids)))
        timestamp = np.array(list(map(int, timestamp)))
        spendtime = np.array(list(map(int, spendtime)))

        ques = np.ones(self.max_seq) * self.n_ques
        ques[:len(ques_ids)] = ques_ids

        concept = np.ones(self.max_seq) * self.n_concept
        concept[:len(ques_ids)] = concept_ids

        st = np.ones(self.max_seq) * 301
        st[:len(spendtime)] = spendtime
        st[len(ques_ids) - 1] = 301

        time_stamp = np.ones(self.max_seq) * 1441
        time_stamp[:len(timestamp)] = timestamp
        time_stamp[len(ques_ids) - 1] = 1441

        time_stamp_tensor = torch.LongTensor(time_stamp)
        a = time_stamp_tensor[:-1]
        sub = torch.cat((torch.LongTensor([0]), a), dim=-1)

        sub[len(ques_ids) - 1] = 1441
        et = time_stamp_tensor - sub
        et[len(ques_ids) - 1:] = 1441
        et = et.clip(0, 1441)


        mask = np.zeros(self.max_seq)
        mask[1:len(answered_correctly) - 1] = 0.9
        mask[len(answered_correctly) - 1] = 1

        labels = np.ones(self.max_seq) * -1
        labels[:len(answered_correctly)] = answered_correctly

        qa = np.ones(self.max_seq) * (self.n_concept * 2 + 1)
        qa[:len(concept_ids)] = concept_ids + answered_correctly * self.n_concept
        qa[len(concept_ids) - 1] = self.n_concept * 2 + 1
        return (torch.LongTensor(qa), torch.LongTensor(ques), torch.LongTensor(labels), torch.FloatTensor(mask),
                torch.LongTensor(concept), torch.LongTensor(st), et)