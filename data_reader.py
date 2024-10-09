import serial
import os
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))
def data_reader():

    # 配置串口参数
    ser = serial.Serial(
        port='COM9',  # 替换为你的ESP32对应的串口号
        baudrate=115200,  # 替换为你的ESP32的波特率
        timeout=1
    )

    save_path = 'train-data\train-data-collect'
    filename = 'TianQiBuCuo'
    window_size = 150

    # 在此修改文件路径
    for file_num in range(1, 105):
        file_path = os.path.join(ROOT_PATH, save_path, filename, rf"\{file_num}.txt")
        # 检查文件是否存在
        if os.path.exists(file_path):
            continue
        else:
            break
        

    # 打开串口
    # ser.open()

    count = 0
    # 打开一个文件用于写入
    with open(file_path, 'w') as f:
        while True:
            if ser.in_waiting:
                line = ser.readline().decode('utf-8')  # 读取一行数据，并解码
                f.write(line)  # 将数据写入文件
                f.flush()  # 确保数据立即写入文件
                print(line, end='')  # 同时在控制台打印数据
                count = count + 1
                if count == 6 * window_size + 20:
                    print('--------------------------------------\n')
                    print('--------------------------------------')
                    break
    return file_path
    # 关闭串口
    # ser.close()
