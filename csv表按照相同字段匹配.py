import pandas as pd
import chardet

def detect_encoding(file_path):
    with open(file_path, 'rb') as file:
        return chardet.detect(file.read())['encoding']

file1_path = r"D:\新建文件夹\Part12_13_19result(1)\Part12_13_19result\Part12-2.csv"         # 目标文件（保留全部行）
file2_path = r"D:\新建文件夹\划分数据\划分数据\7_孙琪Part12至Part13\Part12.csv"       # 来源文件（只取人数列）
output_path = 'Part12_13_19匹配.csv'

enc1 = detect_encoding(file1_path)
enc2 = detect_encoding(file2_path)

df1 = pd.read_csv(file1_path, encoding=enc1)
df2 = pd.read_csv(file2_path, encoding=enc2)

# 只保留文件2中的 OD_link 和人数列（假设列名叫 'num'）
df2_subset = df2[['OD_link', 'num_freq', 'num_freq_t']]

# 左连接，只把人数列匹配到文件1
result = pd.merge(df1, df2_subset, on='OD_link', how='left')

# 保存
result.to_csv(output_path, index=False, encoding='utf-8')

print(f"匹配完成，结果已写入：{output_path}")