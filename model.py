import math
import torch
import json
import os
import torch.nn as nn
import torch.nn.functional as F


class Mpu2TextClassifier(nn.Module):
    def __init__(
        self,
        window_size,
        mpu_channels,
        n_dim,
        n_head,
        dropout,
        cls_num,
        more_than_one_word,
        if_print=False,
    ):
        super().__init__()
        self.conv_head = nn.Sequential(
            nn.Conv1d(
                in_channels=mpu_channels,
                out_channels=n_dim,
                kernel_size=2,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=n_dim,
                out_channels=n_dim,
                kernel_size=window_size // 2,
                stride=window_size // 2,
            ),
        )
        self.cls_token = nn.Parameter(torch.rand(1, 1, n_dim))
        self.pos_embedding = nn.Parameter(torch.rand((1, 120, n_dim)))
        self.block1 = Block(n_dim, n_head, dropout)
        self.block2 = Block(n_dim, n_head, dropout)
        self.cls_head = nn.Linear(n_dim, cls_num)
        self.ctc_criterion = nn.CTCLoss(blank=0, reduction="mean")
        self.ce_criterion = nn.CrossEntropyLoss()
        self.more_than_one_word = more_than_one_word
        self.if_print = if_print
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), r'train-data\ready\word2num.json'),'r') as f:
            word2num_dict=json.load(f)
        self.num2word_dict={value:key for key,value in word2num_dict.items()}

    def forward(self, input_dict):
        x = input_dict["data"]
        pad_mask = input_dict["pad_mask"]
        x = self.conv_head(x).transpose(-1, -2).contiguous()
        x = torch.cat([self.cls_token.repeat(x.shape[0], 1, 1), x], dim=1)
        pad_mask = torch.cat(
            [torch.zeros(size=(pad_mask.shape[0], 1), device=x.device), pad_mask], dim=1
        )
        x = x + self.pos_embedding[:, : x.shape[1], :]
        x = self.block1(x, pad_mask)
        x = self.block2(x, pad_mask)
        logits = self.cls_head(x)
        cls_logits = logits[:, 0, :]
        logits = logits[:, 1:, :]
        decoded = self.ctc_decode(logits.log_softmax(2), pad_mask)
        decoded, acc = self.compute_acc(
            input_dict["label"],
            input_dict["cls_label"],
            decoded,
            torch.argmax(cls_logits, dim=-1),
        )

        return {
            "target": input_dict['label'].tolist(),
            "decoded": decoded,
            "acc": acc,
            "loss": self.ctc_loss(logits, input_dict["label"], pad_mask)
            + self.ce_criterion(cls_logits, input_dict["cls_label"]),
        }

    def compute_acc(self, target_label, target_cls_label, label, cls_label):
        one_word_correct_num = torch.count_nonzero(
            (target_cls_label == cls_label)
            & (target_cls_label != self.more_than_one_word)
        )

        more_than_one_mask = (target_cls_label == self.more_than_one_word) & (
            cls_label == self.more_than_one_word
        )
        target_label_more = target_label[more_than_one_mask]

        tmp = []
        for i in torch.nonzero(more_than_one_mask):
            tmp.append(label[i])

        more_word_correct_num = 0
        for i in range(len(target_label_more)):
            target = target_label_more[i][target_label_more[i] != 0]
            decoded = torch.tensor(tmp[i], dtype=torch.long, device=target.device)
            if target.shape == decoded.shape:
                if torch.equal(target, decoded):
                    more_word_correct_num += 1

        decoded = []
        for i in range(len(label)):
            if cls_label[i] != self.more_than_one_word:
                decoded.append([cls_label[i].item()])
            else:
                decoded.append(label[i])

        if self.if_print:
            for key, value in {
                "target": [' '.join([self.num2word_dict[j] for j in i if j!=0]) for i in target_label.tolist()],
                "decoded": [' '.join([self.num2word_dict[j] for j in i]) for i in decoded],
                "target_cls": target_cls_label.tolist(),
                "cls_label": cls_label.tolist(),
                "one": one_word_correct_num,
                "more": more_word_correct_num,
            }.items():
                print(f"{key}:{value}")

        return decoded, (one_word_correct_num + more_word_correct_num) / len(
            target_label
        )

    def ctc_loss(self, logits, labels, pad_mask):
        logits_prob = logits.log_softmax(2).transpose(0, 1)
        labels = labels.to(torch.long)
        input_length = torch.full(
            (logits_prob.shape[1],), logits_prob.shape[0], dtype=torch.long
        )
        for i in range(input_length.shape[0]):
            pad_mark = (pad_mask[i] == 1).nonzero()
            if pad_mark.shape[0] != 0:
                input_length[i] = pad_mark[0]

        target_length = torch.full(
            (labels.shape[0],), labels.shape[1], dtype=torch.long
        )
        for i in range(target_length.shape[0]):
            pad_mark = (labels[i] == 0).nonzero()
            if pad_mark.shape[0] != 0:
                target_length[i] = pad_mark[0]

        return self.ctc_criterion(logits_prob, labels, input_length, target_length)

    def ctc_decode(self, log_prob, pad_mask):
        index_list = torch.argmax(log_prob, dim=-1).tolist()
        results = []
        for i in range(len(index_list)):
            raw_decoded = index_list[i]
            pad_mark = (pad_mask[i] == 1).nonzero()
            if pad_mark.shape[0] != 0:
                raw_decoded = raw_decoded[: pad_mark[0]]
            combined = []
            for j in range(len(raw_decoded)):
                if j == len(raw_decoded) - 1:
                    if raw_decoded[j] != 0:
                        combined.append(raw_decoded[j])
                    break
                if raw_decoded[j] != raw_decoded[j + 1] and raw_decoded[j] != 0:
                    combined.append(raw_decoded[j])

            results.append(combined)
        return results


class Block(nn.Module):
    def __init__(self, n_dim, n_head, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(n_dim, n_head, dropout)
        self.ff = FeedForward(n_dim, dropout)
        self.pre_ln = nn.LayerNorm(n_dim)
        self.aft_ln = nn.LayerNorm(n_dim)

    def forward(self, x, pad_mask=None):
        x = x + self.attn(self.pre_ln(x), pad_mask)
        x = x + self.ff(self.aft_ln(x))
        return x


class FeedForward(nn.Module):
    def __init__(self, n_dim, dropout):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(n_dim, 4 * n_dim),
            nn.ReLU(),
            nn.Linear(4 * n_dim, n_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layer(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, n_dim, n_head, dropout):
        super().__init__()
        assert n_dim % n_head == 0
        self.n_dim = n_dim
        self.n_head = n_head
        self.qkv = nn.Linear(n_dim, 3 * n_dim)
        self.attn_dropout = nn.Dropout(dropout)
        self.value_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(n_dim, n_dim)

    def forward(self, x, pad_mask):
        if pad_mask == None:
            pad_mask = torch.zeros(x.shape[0], x.shape[1], device=x.device)
        B, T, C = x.shape
        x = self.qkv(x)
        q, k, v = torch.split(x, split_size_or_sections=self.n_dim, dim=-1)
        q = q.view(B, T, self.n_head, -1).transpose(1, 2)
        k = k.view(B, T, self.n_head, -1).transpose(1, 2)
        v = v.view(B, T, self.n_head, -1).transpose(1, 2)
        attn = (q @ k.transpose(-1, -2)) * (1.0 / math.sqrt(k.size(-1)))
        attn = attn.masked_fill(
            pad_mask.unsqueeze(1).unsqueeze(1).to(torch.bool), -torch.inf
        )
        attn = F.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)
        v = attn @ v
        v = v.transpose(1, 2).contiguous().view(B, T, -1)
        v = self.value_dropout(self.proj(v))
        return v
