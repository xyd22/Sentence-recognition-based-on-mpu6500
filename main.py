from prepare import prepare
from train import train
from test import Identifier
from data_reader import data_reader
import os
import shutil
import warnings
warnings.filterwarnings("ignore")

ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
class MAIN():
    def begin(self, mode, TRAIN_FOLDER, MODEL_PATH = 'model.pt'):
        assert mode in ['train', 'test', 'predict', 'collect-data']
        if mode == 'train':
            prepare(ROOT_PATH, TRAIN_FOLDER)
            train(ROOT_PATH, MODEL_PATH)
        
        if mode == 'collect-data':
            save_path = data_reader()
            shutil.copy(save_path, os.path.join(ROOT_PATH, r'TestData\real-time-identify\raw'))
            Identifier().GetResult(mode = mode, MODEL_PATH = 'model.pt')


        if mode != 'train' & mode != 'collect-data':
            identifier = Identifier()
            identifier.GetResult(mode = mode, MODEL_PATH = 'model.pt')

MAIN().begin(mode = 'predict', TRAIN_FOLDER = 'train-data-CN')