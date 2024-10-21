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
    def begin(self, mode, FOLDER, TEST_SAVE_PATH, MODEL_PATH = 'model.pt', RAND_SEED = 42):
        assert mode in ['train', 'test', 'predict', 'collect-data']
        if mode == 'train':
            prepare(ROOT_PATH, FOLDER)
            # train(ROOT_PATH, MODEL_PATH)
            # for i in range(1):
            with open(os.path.join(ROOT_PATH, TEST_SAVE_PATH), 'w') as file:
                train(ROOT_PATH, MODEL_PATH, file, rand_seed = RAND_SEED)
                # train(ROOT_PATH, MODEL_PATH, rand_seed=i)
        
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
            identifier.GetResult(mode = mode, FOLDER_PATH = FOLDER, MODEL_PATH = 'model.pt', SAVE_PATH = TEST_SAVE_PATH)

for i in range(1):
    MAIN().begin(mode = 'train', FOLDER = r'train-data-hzf+patient+xyd_0.7', RAND_SEED = i, TEST_SAVE_PATH = os.path.join(ROOT_PATH, rf'results\person\results{i}.txt'))
    MAIN().begin(mode = 'test', FOLDER = r'TestData\data-xyd-test-0.3', TEST_SAVE_PATH = os.path.join(ROOT_PATH, rf'results\person\xydfew\{i}.txt'))
    # MAIN().begin(mode = 'test', FOLDER = r'TestData\data-hzf-test-0.3', TEST_SAVE_PATH = os.path.join(ROOT_PATH, rf'results\person\hzf\{i}.txt'))
    # MAIN().begin(mode = 'test', FOLDER = r'TestData\data-patient-test-0.3', TEST_SAVE_PATH = os.path.join(ROOT_PATH, rf'results\person\patient\{i}.txt'))
# MAIN().begin(mode = 'predict', FOLDER = r'TestData\real-time-identify')