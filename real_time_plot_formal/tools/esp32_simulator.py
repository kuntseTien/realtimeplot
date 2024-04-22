import socket
import time
import math

# 服務器的地址和端口
server_host = 'localhost'
server_port = 11520

# 創建 socket 對象
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

# 嘗試連接到服務器
try:
    client_socket.connect((server_host, server_port))
    print(f"已連接到服務器 {server_host}:{server_port}")

    # 記錄起始時間
    start_time = time.time()

    # 定義每次發送的數據點數量和間隔時間
    points_per_batch = 20
    batch_interval = 0.02 # 20 毫秒

    while True:
        # 初始化一個批次的數據字符串
        batch_data = ""

        # 獲取當前時間和起始時間的差值
        current_time = time.time() - start_time

        # 生成此批次的所有數據點
        for i in range(points_per_batch):
            # 根據當前時間計算每個數據點的實際時間戳
            t = round(current_time + i * batch_interval / points_per_batch, 5)
            y = math.sin(2 * math.pi * t)  # 生成數據點，頻率為 1000 Hz
            batch_data += f"{t},{round(y, 5)};"  # 添加到批次數據中

        # 發送批次數據
        client_socket.sendall(batch_data.encode('utf-8'))
        print(batch_data)

        # 等待下一批次
        time.sleep(batch_interval)

except Exception as e:
    print(f"連接到服務器時出現錯誤: {e}")

finally:
    # 關閉連接
    client_socket.close()
