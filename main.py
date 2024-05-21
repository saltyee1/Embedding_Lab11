
import os
from machine import Pin, SoftSPI, SPI
from sdcard import SDCard
import math
import ujson  # 導入 ujson 模組來處理 JSON 數據
# SD 卡初始化
spi = SPI(1, sck=Pin(18), mosi=Pin(23), miso=Pin(19))
sd = SDCard(spi, Pin(5))  # 使用適合硬體的引腳
os.mount(sd, '/sd')

def read_ecg_data(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    ecg_data = [int(float(line.strip()) * 1000) for line in lines]
    return ecg_data

def find_peaks_crest(src, min_distance, src_length):
    peaks = []
    for i in range(1, src_length - 1):
        if src[i] > src[i - 1] and src[i] > src[i + 1]:
            is_peak = True
            start = max(i - min_distance, 0)
            end = min(i + min_distance, src_length)
            for j in range(start, end):
                if src[i] < src[j]:
                    is_peak = False
                    break
            if is_peak:
                peaks.append(i)
    return peaks

# Redefine the HRV calculation function to closely follow the C code logic
def calculate_hrv(peaks, sampling_rate):
    intervals = [peaks[i+1] - peaks[i] for i in range(len(peaks)-1)]
    mean_rr = sum(intervals) / len(intervals) / sampling_rate
    var_rr = sum((interval - (mean_rr * sampling_rate)) ** 2 for interval in intervals) / (len(intervals) - 1)
    sd_rr = var_rr ** 0.5 / sampling_rate
    rmssd_rr = (sum((intervals[i] - intervals[i-1]) ** 2 for i in range(1, len(intervals))) / (len(intervals) - 1)) ** 0.5 / sampling_rate
    nn50 = sum(abs(intervals[i] - intervals[i-1]) > (0.05 * sampling_rate) for i in range(1, len(intervals)))
    pnn50 = nn50 / len(intervals) * 100
    return mean_rr, sd_rr, var_rr/100, rmssd_rr, nn50, pnn50


def hrv(file_path):
    ecg_data = read_ecg_data(file_path)
    peaks = find_peaks_crest(ecg_data, 65, len(ecg_data))
    return calculate_hrv(peaks, 100)

# 讀取資料和計算 HRV
def load_matrix_from_file(file_path):
    matrix = []
    # 確保檔案路徑包含 SD 卡的掛載點
    full_path = '/sd/' + file_path  # 假設所有檔案都在 SD 卡的根目錄下
    with open(full_path, 'r') as file:
        for line in file:
            # 移除行尾的換行符和不必要的字符
            processed_line = line.strip().replace('[', '').replace(']', '')
            # 轉換為浮點數列表
            number_list = [float(num) for num in processed_line.split(',') if num.strip()]
            matrix.append(number_list)
    return matrix


def zeros1d(x):  # 1d zero matrix
    z = [0 for i in range(len(x))]
    return z


def add1d(x, y):
    if len(x) != len(y):
        print("Dimention mismatch")
        exit()
    else:
        z = [x[i] + y[i] for i in range(len(x))]
        return z


def relu(x):  # Relu activation function
    
    y = []
    for i in range(len(x)):
        if x[i] >= 0:
            y.append(x[i])
        else:
            y.append(0)

    
    return y

def my_exp(x):
    return math.exp(x)

def my_sum(arr):
    total = 0
    for element in arr:
        total += element
    return total

def softmax(X):
    X=transpose(X)
    exp_X = [[my_exp(x) for x in row] for row in X]  # 計算X中每個元素的指數
    sum_exp_X = [my_sum(row) for row in exp_X]  # 對每行的指數求和

    # 計算 softmax
    softmax_X = [[elem / sum_exp for elem in row] for row, sum_exp in zip(exp_X, sum_exp_X)]
    softmax_X=transpose(softmax_X)
    return softmax_X





def zeros(rows, cols):
    """
    Creates a matrix filled with zeros.
        :param rows: the number of rows the matrix should have
        :param cols: the number of columns the matrix should have
        :return: list of lists that form the matrix
    """
    M = []
    while len(M) < rows:
        M.append([])
        while len(M[-1]) < cols:
            M[-1].append(0.0)

    return M


def transpose(M):
    """
    Returns a transpose of a matrix.
        :param M: The matrix to be transposed
        :return: The transpose of the given matrix
    """
    # Section 1: if a 1D array, convert to a 2D array = matrix
    if not isinstance(M[0], list):
        M = [M]

    # Section 2: Get dimensions
    rows = len(M)
    cols = len(M[0])

    # Section 3: MT is zeros matrix with transposed dimensions
    MT = zeros(cols, rows)

    # Section 4: Copy values from M to it's transpose MT
    for i in range(rows):
        for j in range(cols):
            MT[j][i] = M[i][j]

    return MT


##Sigmoid function

def neuron(x, w, b, activation):  # perform operation on a single neuron and return a 1d array

    tmp = zeros1d(x[0])

    for i in range(len(x)):
        tmp = add1d(tmp, [(float(w[i]) * float(x[i][j])) for j in range(len(x[0]))])

    if activation == "softmax":
        yp=[tmp[i] + b for i in range(len(tmp))]
        
    elif activation == "relu":
        yp = relu([tmp[i] + b for i in range(len(tmp))])
    else:
        print("Invalid activation function--->")

    return yp


def dense(nunit, x, w, b, activation):  # define a single dense layer followed by activation
    res = []
    for i in range(nunit):
        z = neuron(x, w[i], b[i], activation)
        
        res.append(z)
    if activation == "softmax":
      print("SS:",len(res))
      res=softmax(res)
    return res

def print_matrix(M, decimals=3):
    """
    Print a matrix one row at a time
        :param M: The matrix to be printed
    """
    for row in M:
        print([round(x, decimals) + 0 for x in row])


def classification_report(ytrue, ypred):  # print prediction results in terms of metrics and confusion matrix
    tmp = 0
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(ytrue)):
        if ytrue[i] == ypred[i]:  # For accuracy calculation
            tmp += 1
        ##True positive and negative count
        if ytrue[i] == 1 and ypred[i] == 1:  # find true positive
            TP += 1
        if ytrue[i] == 0 and ypred[i] == 0:  # find true negative
            TN += 1
        if ytrue[i] == 0 and ypred[i] == 1:  # find false positive
            FP += 1
        if ytrue[i] == 1 and ypred[i] == 0:  # find false negative
            FN += 1
    accuracy = tmp / len(ytrue)
    conf_matrix = [[TN, FP], [FN, TP]]
    

    print("Accuracy: " + str(accuracy))
    print("Confusion Matrix:")
    print(print_matrix(conf_matrix))

def argmax(arr_x):
  ans=[]
  for i in arr_x:
    maxx=0
    tem=-1
    for j in range(len(arr_x[0])):
      if i[j]>maxx:
        maxx=i[j]
        tem=j
    ans.append(tem)
  return ans


# 定義SD卡路徑和檔案名
drive_path = "/sd/weights"
file_name = "weights_only_70.txt"
file_path = drive_path + "/" + file_name  # 手動建立檔案路徑
# 主程序
if __name__ == "__main__":
    directory_path = '/sd/data_80'
    results = []
    for filename in os.listdir(directory_path):
        if filename.endswith('.csv'):
            full_path = directory_path + '/' + filename
            results.append(hrv(full_path))

    output_file_path = '/sd/hrv_results.txt'
    with open(output_file_path, 'w') as file:
        for i, item in enumerate(results):
            item_str = '[' + ', '.join([f"{x:.4f}" if isinstance(x, float) else str(x) for x in item]) + ']'
            file.write(item_str + (',\n' if i < len(results) - 1 else '\n'))

    print(f"HRV calculations have been saved to {output_file_path}")
# 檢查文件是否存在
if file_name in os.listdir(drive_path):
    print(f"File '{file_name}' found! Proceeding to process data.")

    # 讀取文件內容並依照指定方式分割處理
    with open(file_path, 'r') as file:
        content = file.read()  # 依照空行分割成多個區塊
        blocks = content.split('FFF')
    
    print(len(blocks))


    # 解析每個區塊的JSON內容
    w1 = ujson.loads(blocks[0])  # 第一個區塊為w1
    b1 = ujson.loads(blocks[1])  # 第二個區塊為b1
    w2 = ujson.loads(blocks[2])  # 第三個區塊為w2
    b2 = ujson.loads(blocks[3])  # 第四個區塊為b2
else:
    print(f"File '{file_name}' not found.")


#Transpose all weight matrix
w1 = transpose(w1)
w2 = transpose(w2)

# 修正後的文件路徑
X_test_file_path = 'hrv_results.txt'  # 更新檔名以符合實際輸出的檔名

# 載入測試數據
Xtest = load_matrix_from_file(X_test_file_path)
print("Loaded test data:", Xtest)




ytrue = [1]*40 + [0]*40
print(len(ytrue))
Xtest=transpose(Xtest)
#Transpose Xtest before feeding to NN
yout1 = dense(70, Xtest, w1, b1, 'relu') #input layer with 4 neuron

ypred = dense(2, yout1, w2, b2,'softmax') #output layer
ypred=transpose(ypred)
print(ypred)
ypred_class=argmax(ypred)

print(ypred_class)
print(classification_report(ytrue,ypred_class))
