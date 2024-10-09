import os
import torch
import re
txt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), r'train-data\train-data-CN\FanShen\1_raw.txt')
data_all = []
with open(txt_path, "r") as f:
    cnt = 0
    for line in f:
        line_data = line.strip()
        if len(line_data) <= 30:
            cnt = cnt + 1
            continue
        if cnt == 3:
            continue
        data_num = re.findall(r"(-?\d+)", line_data[0 : len(line_data)])
        data_num = [float(data_num[i]) for i in range(len(data_num))]
        data_all.append(data_num)
# print(torch.tensor(data_all, dtype=torch.float32))