from prepare import prepare
from train import train
from test import Identifier
import os
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
class MAIN():
    def begin(self, mode, TRAIN_FOLDER, MODEL_PATH = 'model.pt'):
        assert mode in ['train', 'test', 'predict']
        if mode == 'train':
            prepare(ROOT_PATH, TRAIN_FOLDER)
            train(ROOT_PATH, MODEL_PATH)

        if mode != 'train':
            identifier = Identifier()
            identifier.GetResult(mode = mode, MODEL_PATH = 'model.pt')

MAIN().begin(mode = 'predict', TRAIN_FOLDER = 'train-data-CN')