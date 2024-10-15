from prepare import prepare
from train import train
from test import Identifier
from data_reader import data_reader
from data_raw_processor import data_raw_process
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
            for i in range(20):
                train(ROOT_PATH, MODEL_PATH, rand_seed=i)
            
        if mode == 'collect-data':
            save_path = data_reader()
            filename = os.path.split(os.path.dirname(save_path))[1]
            sample_num = int(os.path.splitext(os.path.basename(save_path))[0])
            data_raw_process(filename, 80, sample_num, sample_num + 1, ROOT_PATH = os.path.dirname(os.path.dirname(save_path)))
            shutil.copy(os.path.join(os.path.dirname(save_path), f'{sample_num}_raw.txt'), os.path.join(ROOT_PATH, r'TestData\real-time-identify\raw'))
            identifier = Identifier()
            identifier.GetResult(mode = 'predict', MODEL_PATH = 'model.pt')
            os.unlink(os.path.join(ROOT_PATH, rf'TestData\real-time-identify\raw\{sample_num}_raw.txt'))


        if mode != 'train' and mode != 'collect-data':
            identifier = Identifier()
            identifier.GetResult(mode = mode, MODEL_PATH = 'model.pt')

MAIN().begin(mode = 'train', TRAIN_FOLDER = 'train-data-xyd')