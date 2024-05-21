# 定義SD卡路徑和檔案名
drive_path = "/sd/weights"
file_name = "weights_only_40.txt"
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
yout1 = dense(40, Xtest, w1, b1, 'relu') #input layer with 4 neuron

ypred = dense(2, yout1, w2, b2,'softmax') #output layer
ypred=transpose(ypred)
print(ypred)
ypred_class=argmax(ypred)

print(ypred_class)
print(classification_report(ytrue,ypred_class))
