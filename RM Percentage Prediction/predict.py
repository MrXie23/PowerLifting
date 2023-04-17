import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

# 读取训练数据
data = pd.read_csv("training_data.csv")

# 分离输入和输出
X = data[["reps", "rpe"]]
y = data["percentage_1rm"]

# 创建回归模型
reg = LinearRegression().fit(X, y)

def predict_percentage(reps, rpe):
    # 预测1RM的百分比
    percentage = reg.predict(np.array([[reps, rpe]]))[0]
    return percentage

def round_down(num):
    return (num // 2.5) * 2.5


# 测试数据
reps = 12
rpe = 9
one_rep_max = 140

# 调用预测函数
percentage = predict_percentage(reps, rpe)

# 计算杠铃片配重
closest_weight = round_down(one_rep_max*percentage)


print("1RM Percentage:", round(percentage,2)*100,"%")
print("Weights:", closest_weight,"kg")
