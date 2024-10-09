import json
import re
import torch
import torch.utils
from model import Mpu2TextClassifier
from torch.nn.utils.rnn import pad_sequence
import os
from dataset import normalize
def read_txt_to_tensor(txt_path):
    data_all = []
    with open(txt_path, "r") as f:
        for line in f:
            line_data = line.strip()
            if len(line_data) <= 30:
                continue
            data_num = re.findall(r"(-?\d+)", line_data[0 : len(line_data)])
            data_num = [float(data_num[i]) for i in range(len(data_num))]
            data_all.append(data_num)
    return torch.tensor(data_all, dtype=torch.float32)

def batched_bincount(x, max_value, dim=2):
    target = torch.zeros(
        x.shape[0], x.shape[1], max_value + 1, dtype=x.dtype, device=x.device
    )
    values = torch.ones_like(x)
    target.scatter_add_(dim, x, values)
    return target

def collate_fn(batch):
    label = pad_sequence([i["label"] for i in batch], batch_first=True, padding_value=0)
    data = pad_sequence(
        [i["data"].transpose(0, 1) for i in batch], batch_first=True, padding_value=0
    ).transpose(1, 2)
    pad_length = (data.shape[2] // 20 + 1) * 20 - data.shape[2]
    data = torch.cat(
        [data, torch.zeros(size=(data.shape[0], data.shape[1], pad_length))], dim=-1
    )
    pad_mask = (data == 0)[:, 0, :].reshape(data.shape[0], -1, 20)
    pad_mask = torch.argmax(
        batched_bincount(pad_mask.to(torch.int64), max_value=1), dim=-1
    )
    return {
        "data": data,
        "label": label,
        "cls_label": torch.hstack([i["cls_label"] for i in batch]),
        "pad_mask": pad_mask,
    }
class Identifier():
    def GetResult(self, mode, MODEL_PATH):
        assert mode in ['test', 'predict']
        ROOT_PATH = os.path.dirname(os.path.abspath(__file__))       # 项目根目录

        if mode == 'test':
            TEST_PATH = os.path.join(ROOT_PATH, r'TestData\cnTest')      # 测试项目路径
        if mode == 'predict':
            TEST_PATH = os.path.join(ROOT_PATH, r'TestData\real-time-identify')      # 测试项目路径

        RAW_WORD_PATH = os.path.join(TEST_PATH, r'raw')              # 未处理的测试数据存放位置
        READY_WORD_PATH = os.path.join(TEST_PATH, r'ready')          # 处理好的测试数据存放位置
        os.makedirs(RAW_WORD_PATH, exist_ok=True)
        os.makedirs(READY_WORD_PATH, exist_ok=True)
        with open(os.path.join(ROOT_PATH, r"train-data\ready\word2num.json"), "r") as f:
            word2num_dict = json.load(f)
        with open(os.path.join(ROOT_PATH, r"train-data\ready\num2word.json"), "r") as f:
            num2word_dict = json.load(f)
        num2word_dict = {int(key):value for key, value in num2word_dict.items()}
        sample_json = []
        if mode == 'test':
            for i in os.listdir(RAW_WORD_PATH):
                for j in os.listdir(os.path.join(RAW_WORD_PATH, i)):
                    os.makedirs(os.path.join(READY_WORD_PATH, i), exist_ok=True)
                    if j[-8:] != '_raw.txt':
                        continue
                    torch.save(read_txt_to_tensor(os.path.join(RAW_WORD_PATH, i, j)), os.path.join(READY_WORD_PATH, i, j[:-4] + '.pt'))
                    sample_json.append(
                        {
                            "path": os.path.join(READY_WORD_PATH, i, j[:-4] + '.pt'),
                            "label": [word2num_dict[i]],
                            "cls_label": word2num_dict[i],
                        }
                    )
        if mode == 'predict':
            for i in os.listdir(RAW_WORD_PATH):
                if i[-8:] != '_raw.txt':
                    continue
                torch.save(read_txt_to_tensor(os.path.join(RAW_WORD_PATH, i)), os.path.join(READY_WORD_PATH, i[:-4] + '.pt'))
                sample_json.append(
                    {
                        "path": os.path.join(READY_WORD_PATH, i[:-4] + '.pt'),
                        "label": [],
                        "cls_label": 1,
                    }
                )
        with open(os.path.join(READY_WORD_PATH, 'test.json'),'w') as f:
            json.dump(sample_json,f)
        # 加载模型
        model = Mpu2TextClassifier(
            window_size=20,
            mpu_channels=36,
            n_dim=128,
            n_head=4,
            dropout=0.5,
            cls_num=len(word2num_dict),
            more_than_one_word=word2num_dict["<more_than_one_word>"],
            if_print=False
        )
        model.load_state_dict(torch.load(MODEL_PATH, weights_only = False))  # 加载模型参数
        model.eval()  # 将模型设置为评估模式




    # 这里为了专门针对一个样本
        class MpuDataset(torch.utils.data.Dataset):
            def __init__(self):
                with open(os.path.join(READY_WORD_PATH, 'test.json'),'r') as f:
                    self.data=json.load(f)

            def __len__(self):
                return len(self.data)

            def __getitem__(self,idx):
                return {
                    'data':normalize(torch.load(self.data[idx]['path'], weights_only = False)),
                    'label':torch.tensor(self.data[idx]['label'],dtype=torch.long),
                    'cls_label':torch.tensor(self.data[idx]['cls_label'],dtype=torch.long)
                }

        test_dataset = MpuDataset()
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=50, shuffle=False, collate_fn=collate_fn
        )

        output_result = []
        for input_dict in test_loader:
            with torch.no_grad():  # 在预测时不需要计算梯度
                output_dict = model(input_dict)
                output_result.append(output_dict)  # 使用解包运算符合并字典

        class Result:
            def __init__(self, sampleNum = 0, correct = 0, acc = 0):
                self.sampleNum = sampleNum
                self.correct = correct
                self.acc = acc
            def AddSample(self, iscorrect):
                self.sampleNum = self.sampleNum + 1
                self.correct = self.correct + int(iscorrect)
                self.acc = self.correct / self.sampleNum
            def __str__(self):
                return f"sampleNum = {self.sampleNum}, correct = {self.correct}, acc = {self.acc:.2f}"

        if mode == 'test':
            result = {}
            for i in output_result:
                for j in range(0, len(i['target'])):
                    res = ''.join(num2word_dict[k] for k in i['target'][j])
                    if(res not in result):
                        result[res] = Result()
                    result[res].AddSample(i['target'][j] == i['decoded'][j])

            for key, value in result.items():
                print(f"{key}:\t{value}")
        if mode == 'predict':
            result = {}
            for i in output_result:
                for j in range(0, len(i['decoded'])):
                    res = ''.join(num2word_dict[k] for k in i['decoded'][j])
                    print(res)