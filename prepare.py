import re
import os
import json
import random
import torch


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

def augment(
    annot_json, target_data_num, max_seq_length, min_seq_length, more_than_one_word, READY_PATH
):
    seq_range = [i for i in range(min_seq_length, max_seq_length)]
    READY_AUGMENTED_PATH = os.path.join(READY_PATH, "augment")
    os.makedirs(READY_AUGMENTED_PATH, exist_ok=True)
    augmented_annot_json = []
    for i in range(target_data_num):
        seq_length = random.choice(seq_range)
        patch = random.sample(annot_json, seq_length)
        data = torch.hstack([torch.load(j["path"]) for j in patch])
        save_path = os.path.join(READY_AUGMENTED_PATH, f"{i}.pt")
        torch.save(data, save_path)
        label = []
        for j in patch:
            label += j["label"]
        augmented_annot_json.append(
            {"path": save_path, "label": label, "cls_label": more_than_one_word}
        )

    return augmented_annot_json


def split_by_random(data):
    N = len(data)
    random.shuffle(data)
    return data[: int(N * 0.8)], data[int(N * 0.8) :]

def prepare(ROOT_PATH, TRAIN_FOLDER):
    RAW_PATH = os.path.join(ROOT_PATH, r"train-data", TRAIN_FOLDER)
    READY_PATH = os.path.join(ROOT_PATH, r"train-data\ready")
    os.makedirs(READY_PATH, exist_ok=True)
    random.seed(42)
    word_list = [
        i
        for i in os.listdir(RAW_PATH)
        if len(i.split(" ")) == 1 and os.path.isdir(os.path.join(RAW_PATH, i))
    ]
    sentence_list = [
        i
        for i in os.listdir(RAW_PATH)
        if len(i.split(" ")) > 1 and os.path.isdir(os.path.join(RAW_PATH, i))
    ]

    word2num_dict = {word: i + 1 for i, word in enumerate(word_list)}
    word2num_dict["<blank>"] = 0
    word2num_dict["<more_than_one_word>"] = len(word2num_dict)

    num2word_dict = {i + 1:word for i, word in enumerate(word_list)}
    num2word_dict[0] = "<blank>"
    num2word_dict[len(num2word_dict)] = "<more_than_one_word>"

    with open(os.path.join(READY_PATH, "word2num.json"), "w") as f:
        json.dump(word2num_dict, f)

    with open(os.path.join(READY_PATH, "num2word.json"), "w") as f:
        json.dump(num2word_dict, f)

    word_annot_json = []
    for word in word_list:
        RAW_WORD_PATH = os.path.join(RAW_PATH, word)
        READY_WORD_PATH = os.path.join(READY_PATH, "word", word)
        os.makedirs(READY_WORD_PATH, exist_ok=True)
        counter = 0
        for filename in os.listdir(RAW_WORD_PATH):
            if "raw.txt" in filename:
                data = read_txt_to_tensor(os.path.join(RAW_WORD_PATH, filename))
                save_path = os.path.join(READY_WORD_PATH, f"{counter}.pt")
                torch.save(data, save_path)
                word_annot_json.append(
                    {
                        "path": save_path,
                        "label": [word2num_dict[word]],
                        "cls_label": word2num_dict[word],
                    }
                )
                counter += 1


    # sentence_annot_json = []
    # for sentence in sentence_list:
    #     RAW_SENTENCE_PATH = os.path.join(RAW_PATH, sentence)
    #     READY_SENTENCE_PATH = os.path.join(READY_PATH, "sentence", sentence)
    #     os.makedirs(READY_SENTENCE_PATH, exist_ok=True)
    #     counter = 0
    #     for filename in os.listdir(RAW_SENTENCE_PATH):
    #         if "raw.txt" in filename:
    #             data = read_txt_to_tensor(os.path.join(RAW_SENTENCE_PATH, filename))
    #             save_path = os.path.join(READY_SENTENCE_PATH, f"{counter}.pt")
    #             torch.save(data, save_path)
    #             sentence_annot_json.append(
    #                 {
    #                     "path": save_path,
    #                     "label": [
    #                         word2num_dict[word.lower()]
    #                         for word in re.split(r",|\s", sentence)
    #                         if word
    #                     ],
    #                     "cls_label": word2num_dict["<more_than_one_word>"],
    #                 }
    #             )
    #             counter += 1

                

    # augmented_annot_json = augment(
    #     # word_annot_json + sentence_annot_json,
    #     word_annot_json,
    #     target_data_num=3000,
    #     max_seq_length=20,
    #     min_seq_length=10,
    #     more_than_one_word=word2num_dict["<more_than_one_word>"],
    #     READY_PATH=READY_PATH,
    # )

    word_train,word_test=split_by_random(word_annot_json)
    with open(os.path.join(READY_PATH,'word','train.json'),'w') as f:
        json.dump(word_train,f)
    with open(os.path.join(READY_PATH,'word','test.json'),'w') as f:
        json.dump(word_test,f)

    # sent_train,sent_test=split_by_random(sentence_annot_json)
    # with open(os.path.join(READY_PATH,'sentence','train.json'),'w') as f:
    #     json.dump(sent_train,f)
    # with open(os.path.join(READY_PATH,'sentence','test.json'),'w') as f:
    #     json.dump(sent_test,f)

    # augment_train,augment_test=split_by_random(augmented_annot_json)
    # with open(os.path.join(READY_PATH,'augment','train.json'),'w') as f:
    #     json.dump(augment_train,f)
    # with open(os.path.join(READY_PATH,'augment','test.json'),'w') as f:
    #     json.dump(augment_test,f)

    with open(os.path.join(READY_PATH,'train.json'),'w') as f:
        # json.dump(augment_train+word_train+sent_train,f)
        json.dump(word_train,f)
    with open(os.path.join(READY_PATH,'test.json'),'w') as f:
        # json.dump(augment_test+word_test+sent_test,f)
        json.dump(word_test,f)