import numpy as np
import pandas as pd

def entropy_weight_from_file(file_path):
    """
    参数:
    file_path: 字符串，数据文件路径（支持csv、xlsx格式）
    返回:
    weights_df: DataFrame，包含指标名称和对应的权重
    """
    # 根据文件扩展名读取数据
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("不支持的文件格式，请使用csv或xlsx格式")
    
    # 确保数据为数值型
    data = data.select_dtypes(include=[np.number])
    if data.empty:
        raise ValueError("数据中没有有效的数值列")
    
    # 转换为numpy数组
    data_np = data.values.astype(float)
    n, m = data_np.shape  # n为样本数，m为指标数
    
    # 计算指标比重
    sum_cols = np.sum(data_np, axis=0)
    proportion = np.zeros_like(data_np)
    for j in range(m):
        if sum_cols[j] == 0:
            proportion[:, j] = 0
        else:
            proportion[:, j] = data_np[:, j] / sum_cols[j]
    
    # 计算信息熵
    entropy = np.zeros(m)
    for j in range(m):
        for i in range(n):
            if proportion[i, j] > 0:
                entropy[j] -= proportion[i, j] * np.log(proportion[i, j])
        if n > 1:
            entropy[j] /= np.log(n)  # 归一化熵值
    
    # 计算权重
    redundancy = 1 - entropy
    if np.sum(redundancy) == 0:
        weights = np.ones(m) / m  # 均匀分配权重
    else:
        weights = redundancy / np.sum(redundancy)
    
    # 构建结果DataFrame
    weights_df = pd.DataFrame({
        '指标': data.columns,
        '权重': weights
    }).sort_values(by='权重', ascending=False)
    
    return weights_df
# 使用方法
if __name__ == "__main__":
    # 请将此处替换为你的数据文件路径
    file_path = "your_data.csv"  # 或 "your_data.xlsx"
    try:
        # 计算权重
        result = entropy_weight_from_file(file_path)
        
        # 输出结果
        print("指标权重计算结果（按权重降序排列）：")
        print("=" * 40)
        print(result.to_string(index=False))  # 不显示索引
    except Exception as e:
        print(f"发生错误：{str(e)}")
