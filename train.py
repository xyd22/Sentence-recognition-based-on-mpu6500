import json
import random
import torch
import torch.utils
import torch.nn as nn
import torch.optim as optim
import os
import numpy as np
from tqdm import tqdm
from dataset import NeckMpuDataset
from model import Mpu2TextClassifier
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    label = pad_sequence([i["label"] for i in batch], batch_first=True, padding_value=0)
    data = pad_sequence(
        [i["data"].transpose(0, 1) for i in batch], batch_first=True, padding_value=0
    ).transpose(1, 2)
    pad_length = (data.shape[2] // 20 + 1) * 20 - data.shape[2]
    data = torch.cat(
        [data, torch.zeros(size=(data.shape[0], data.shape[1], pad_length))], dim=-1
    )

    return {
        "data": data,
        "label": label,
        "cls_label": torch.hstack([i["cls_label"] for i in batch]),
        "length": torch.hstack([i["length"] for i in batch]),
    }

def train(ROOT_PATH, MODEL_PATH, file, rand_seed = 42):
    random.seed(rand_seed)
    torch.manual_seed(rand_seed)
    torch.cuda.manual_seed(rand_seed)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(os.path.join(ROOT_PATH, r"train-data\\ready\\word2num.json"), "r") as f:
        word2num_dict = json.load(f)

    num_epochs = 20
    batch_size = 32


    train_dataset = NeckMpuDataset(mode="train")
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    valid_dataset = NeckMpuDataset(mode="valid")
    valid_loader = torch.utils.data.DataLoader(
    valid_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    test_dataset = NeckMpuDataset(mode="test")
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )

    model = Mpu2TextClassifier(
        window_size=20,
        mpu_channels=30,
        n_dim=128,
        n_head=4,
        dropout=0.5,
        cls_num=len(word2num_dict),
        more_than_one_word=word2num_dict["<more_than_one_word>"],
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

    loss_history = []
    acc_history = []
    for epoch in range(num_epochs):
        model.train()
        mean_loss = 0.0
        mean_acc = 0.0
        with tqdm(train_loader, unit="batch") as tepoch:
            for input_dict in tepoch:
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].to(DEVICE, non_blocking=True)
                tepoch.set_description(f"Epoch {epoch + 1}")
                optimizer.zero_grad()
                output_dict = model(input_dict)
                loss = output_dict["loss"]
                acc = output_dict["acc"]
                mean_loss += loss.item()
                mean_acc += acc.item()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
                optimizer.step()
                tepoch.set_postfix(acc=acc.item())

        mean_loss /= len(train_loader)
        mean_acc /= len(train_loader)
        loss_history.append(mean_loss)
        acc_history.append(mean_acc)

        mean_loss = 0.0
        mean_acc = 0.0
        # 验证阶段
        model.eval()  # 将模型设置为评估模式
        with torch.no_grad():  # 确保不会计算梯度，因为我们不在验证集上训练
            for input_dict in valid_loader:
                for key in input_dict.keys():
                    if isinstance(input_dict[key], torch.Tensor):
                        input_dict[key] = input_dict[key].to(DEVICE, non_blocking=True)
                output_dict = model(input_dict)
                loss = output_dict["loss"]
                acc = output_dict["acc"]
                mean_loss += loss.item()
                mean_acc += acc.item()

        # 计算测试集上的平均损失和准确率
        mean_loss /= len(valid_loader)
        mean_acc /= len(valid_loader)

        # 打印测试结果    
        # with open(rf"C:\Users\hp\Desktop\Sentence-recognition-based-on-mpu6500\Sentence-recognition-based-on-mpu6500\results\loss&acc\{rand_seed}.txt", 'w') as file:
        file.write(f"Test Epoch {epoch + 1}, Loss: {mean_loss:.4f}, Accuracy: {mean_acc:.4f}\n")
        file.flush()
        print(f"Test Epoch {epoch + 1}, Loss: {mean_loss:.4f}, Accuracy: {mean_acc:.4f}")

    mean_loss = np.zeros((1, 21))
    mean_acc = np.zeros((1, 21))
    loss_total = 0.0
    acc_total = 0.0
    count_length = np.zeros((1, 21))
    # 测试阶段
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():  # 确保不会计算梯度，因为我们不在测试集上训练
        for input_dict in test_loader:
            for key in input_dict.keys():
                if isinstance(input_dict[key], torch.Tensor):
                    input_dict[key] = input_dict[key].to(DEVICE, non_blocking=True)
            output_dict = model(input_dict)
            loss = output_dict["loss"]
            acc = output_dict["acc"]
            length = output_dict["length"]
            mean_loss[0][length] += loss.item()
            loss_total += loss.item()
            mean_acc[0][length] += acc.item()
            acc_total += acc.item()
            count_length[0][length] += 1

    # with open(rf"C:\Users\hp\Desktop\Sentence-recognition-based-on-mpu6500\Sentence-recognition-based-on-mpu6500\results\loss&acc\{rand_seed}.txt", 'w') as file:
        # 计算测试集上的平均损失和准确率
        for i in range(21):
            mean_loss[0][i] /= count_length[0][i]
            mean_acc[0][i] /= count_length[0][i]
        # 打印保存测试结果
            # file.write(f"Test on Length {i}, Loss: {mean_loss[0][i]:.4f}, Accuracy: {mean_acc[0][i]:.4f}\n")
            # file.flush()
            print(f"Test on Length {i}, Loss: {mean_loss[0][i]:.4f}, Accuracy: {mean_acc[0][i]:.4f}")

        loss_total /= len(test_loader)
        acc_total /= len(test_loader)
        file.write(f"Test Result, Loss: {loss_total:.4f}, Accuracy: {acc_total:.4f}\n")
        file.flush()
        print(f"Test Result, Loss: {loss_total:.4f}, Accuracy: {acc_total:.4f}")


    torch.save(model.state_dict(), MODEL_PATH)
    torch.save(loss_history, "loss.pt")
    torch.save(acc_history, "acc.pt")
