import re
import numpy as np
import os

def data_process(mpu_data, length):
    # 无偏
    mpu_data_0 = int(mpu_data[0])
    for i in range(length):
        mpu_data[i] = int(mpu_data[i]) - mpu_data_0
    mpu_data = mpu_data[0 : length]
    mpu_data = [str(mpu_data[i]) for i in range(length)]
    return mpu_data

# filename = 'Drink water'
def data_raw_process(filename, total_num, start_num, end_num, ROOT_PATH = rf"C:\Users\hp\Desktop\Sentence-recognition-based-on-mpu6500\Sentence-recognition-based-on-mpu6500\train-data\new-data"):
    for count in range(start_num, end_num):
        # total_num = 180
        mpu0_ax = []
        mpu0_ay = []
        mpu0_az = []
        mpu0_gx = []
        mpu0_gy = []
        mpu0_gz = []
        mpu1_ax = []
        mpu1_ay = []
        mpu1_az = []
        mpu1_gx = []
        mpu1_gy = []
        mpu1_gz = []
        mpu2_ax = []
        mpu2_ay = []
        mpu2_az = []
        mpu2_gx = []
        mpu2_gy = []
        mpu2_gz = []
        mpu3_ax = []
        mpu3_ay = []
        mpu3_az = []
        mpu3_gx = []
        mpu3_gy = []
        mpu3_gz = []
        mpu4_ax = []
        mpu4_ay = []
        mpu4_az = []
        mpu4_gx = []
        mpu4_gy = []
        mpu4_gz = []
        mpu5_ax = []
        mpu5_ay = []
        mpu5_az = []
        mpu5_gx = []
        mpu5_gy = []
        mpu5_gz = []
        file_path = os.path.join(ROOT_PATH, filename, f'{count}.txt')
        with open(file_path, 'r') as f:
            for line in f:
                line_data = line.strip()
                if len(line_data) <= 2:
                    continue
                first_char = line_data[0]
                if not first_char.isdigit():
                    continue
                data_num = re.findall(r'(-?\d+)', line_data[3:len(line_data)])
                for i in range(len(data_num)):
                    data_num[i] = int(data_num[i])
                    if data_num[i] >= 32768:
                        data_num[i] = data_num[i] - 65536
                    data_num[i] = str(data_num[i])
                if line_data[0] == '0':
                    mpu0_ax.append(data_num[0])
                    mpu0_ay.append(data_num[1])
                    mpu0_az.append(data_num[2])
                    mpu0_gx.append(data_num[3])
                    mpu0_gy.append(data_num[4])
                    mpu0_gz.append(data_num[5])
                if line_data[0] == '1':
                    mpu1_ax.append(data_num[0])
                    mpu1_ay.append(data_num[1])
                    mpu1_az.append(data_num[2])
                    mpu1_gx.append(data_num[3])
                    mpu1_gy.append(data_num[4])
                    mpu1_gz.append(data_num[5])
                if line_data[0] == '2':
                    mpu2_ax.append(data_num[0])
                    mpu2_ay.append(data_num[1])
                    mpu2_az.append(data_num[2])
                    mpu2_gx.append(data_num[3])
                    mpu2_gy.append(data_num[4])
                    mpu2_gz.append(data_num[5])
                if line_data[0] == '3':
                    mpu3_ax.append(data_num[0])
                    mpu3_ay.append(data_num[1])
                    mpu3_az.append(data_num[2])
                    mpu3_gx.append(data_num[3])
                    mpu3_gy.append(data_num[4])
                    mpu3_gz.append(data_num[5])
                if line_data[0] == '4':
                    mpu4_ax.append(data_num[0])
                    mpu4_ay.append(data_num[1])
                    mpu4_az.append(data_num[2])
                    mpu4_gx.append(data_num[3])
                    mpu4_gy.append(data_num[4])
                    mpu4_gz.append(data_num[5])
                if line_data[0] == '5':
                    mpu5_ax.append(data_num[0])
                    mpu5_ay.append(data_num[1])
                    mpu5_az.append(data_num[2])
                    mpu5_gx.append(data_num[3])
                    mpu5_gy.append(data_num[4])
                    mpu5_gz.append(data_num[5])
        
        # 数据处理
        mpu0_ax = data_process(mpu0_ax, total_num)
        mpu0_ay = data_process(mpu0_ay, total_num)
        mpu0_az = data_process(mpu0_az, total_num)
        mpu0_gx = data_process(mpu0_gx, total_num)
        mpu0_gy = data_process(mpu0_gy, total_num)
        mpu0_gz = data_process(mpu0_gz, total_num)
        mpu1_ax = data_process(mpu1_ax, total_num)
        mpu1_ay = data_process(mpu1_ay, total_num)
        mpu1_az = data_process(mpu1_az, total_num)
        mpu1_gx = data_process(mpu1_gx, total_num)
        mpu1_gy = data_process(mpu1_gy, total_num)
        mpu1_gz = data_process(mpu1_gz, total_num)
        mpu2_ax = data_process(mpu2_ax, total_num)
        mpu2_ay = data_process(mpu2_ay, total_num)
        mpu2_az = data_process(mpu2_az, total_num)
        mpu2_gx = data_process(mpu2_gx, total_num)
        mpu2_gy = data_process(mpu2_gy, total_num)
        mpu2_gz = data_process(mpu2_gz, total_num)
        mpu3_ax = data_process(mpu3_ax, total_num)
        mpu3_ay = data_process(mpu3_ay, total_num)
        mpu3_az = data_process(mpu3_az, total_num)
        mpu3_gx = data_process(mpu3_gx, total_num)
        mpu3_gy = data_process(mpu3_gy, total_num)
        mpu3_gz = data_process(mpu3_gz, total_num)
        mpu4_ax = data_process(mpu4_ax, total_num)
        mpu4_ay = data_process(mpu4_ay, total_num)
        mpu4_az = data_process(mpu4_az, total_num)
        mpu4_gx = data_process(mpu4_gx, total_num)
        mpu4_gy = data_process(mpu4_gy, total_num)
        mpu4_gz = data_process(mpu4_gz, total_num)
        mpu5_ax = data_process(mpu5_ax, total_num)
        mpu5_ay = data_process(mpu5_ay, total_num)
        mpu5_az = data_process(mpu5_az, total_num)
        mpu5_gx = data_process(mpu5_gx, total_num)
        mpu5_gy = data_process(mpu5_gy, total_num)
        mpu5_gz = data_process(mpu5_gz, total_num)

        # total_num = len(mpu0_ax)
        # 写文件    
        target_path = os.path.join(ROOT_PATH, filename, f'{count}_raw.txt')
        with open(target_path, 'w') as file:
            file.write("mpu0:\n")
            file.write(' '.join(mpu0_ax[0:total_num]))  # 将数据写入文件，只保留前total_num个数据
            file.write('\n')
            file.flush()  # 确保数据立即写入文件
            file.write(' '.join(mpu0_ay[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu0_az[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu0_gx[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu0_gy[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu0_gz[0:total_num]))
            file.flush()
            file.write('\n')

            file.write("mpu1:\n")
            file.write(' '.join(mpu1_ax[0:total_num]))  # 将数据写入文件
            file.write('\n')
            file.flush()  # 确保数据立即写入文件
            file.write(' '.join(mpu1_ay[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu1_az[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu1_gx[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu1_gy[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu1_gz[0:total_num]))
            file.flush()
            file.write('\n')

            file.write("mpu2:\n")
            file.write(' '.join(mpu2_ax[0:total_num]))  # 将数据写入文件
            file.write('\n')
            file.flush()  # 确保数据立即写入文件
            file.write(' '.join(mpu2_ay[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu2_az[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu2_gx[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu2_gy[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu2_gz[0:total_num]))
            file.flush()
            file.write('\n')
            
            file.write("mpu3:\n")
            file.write(' '.join(mpu3_ax[0:total_num]))  # 将数据写入文件
            file.write('\n')
            file.flush()  # 确保数据立即写入文件
            file.write(' '.join(mpu3_ay[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu3_az[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu3_gx[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu3_gy[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu3_gz[0:total_num]))
            file.flush()
            file.write('\n')

            file.write("mpu4:\n")
            file.write(' '.join(mpu4_ax[0:total_num]))  # 将数据写入文件
            file.write('\n')
            file.flush()  # 确保数据立即写入文件
            file.write(' '.join(mpu4_ay[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu4_az[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu4_gx[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu4_gy[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu4_gz[0:total_num]))
            file.flush()
            file.write('\n')

            file.write("mpu5:\n")
            file.write(' '.join(mpu5_ax[0:total_num]))  # 将数据写入文件
            file.write('\n')
            file.flush()  # 确保数据立即写入文件
            file.write(' '.join(mpu5_ay[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu5_az[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu5_gx[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu5_gy[0:total_num]))
            file.flush()
            file.write('\n')
            file.write(' '.join(mpu5_gz[0:total_num]))
            file.flush()
            file.write('\n')

data_raw_process("WoXiangChiFan", 100, 62, 67)