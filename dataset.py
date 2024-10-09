import os
import json
import torch
import torch.utils
import torch.nn as nn
def normalize(mpu_data):
    mean=torch.mean(mpu_data,dim=1,keepdim=True)
    std=torch.std(mpu_data,dim=1,keepdim=True)
    return (mpu_data-mean)/std

class NeckMpuDataset(torch.utils.data.Dataset):
    def __init__(self,mode,level=None):
        assert mode in ['train','test']
        ROOT_PATH=os.path.join(os.path.dirname(os.path.abspath(__file__)), r'train-data\ready')
        JSON_PATH=os.path.join(ROOT_PATH,f'{mode}.json')
        if level!=None:
            JSON_PATH=os.path.join(ROOT_PATH,level,f'{mode}.json')
        with open(JSON_PATH,'r') as f:
            self.data=json.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return {
            'data':normalize(torch.load(self.data[idx]['path'])),
            'label':torch.tensor(self.data[idx]['label'],dtype=torch.long),
            'cls_label':torch.tensor(self.data[idx]['cls_label'],dtype=torch.long)
        }