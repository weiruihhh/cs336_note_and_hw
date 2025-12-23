"""第一步先解析json数据"""
import json
from collections import defaultdict

isojson_path = 'data/isoflops_curves.json'

with open(isojson_path, 'r') as f:
    data = json.load(f)
# print(type(data)) List[dict[str, float]]

grouped_data = defaultdict(list)
#把原始的compute_budget的value现在作为 key来重整dict[float, List[dict[str, float]]]。
for item in data:
    key = item.get('compute_budget')
    grouped_data[key].append(item)
#每个key内都是对应组的List了。

sort_key = 'final_loss'
for key in grouped_data:
    grouped_data[key].sort(
        key=lambda x: x.get(sort_key), # 对某一个List[dict[str, float]]内某一个str进行排序
        reverse=False
    )
# print(compute_budget_value)

"""第二步找出每一个C下对应最小的Loss，并找出对应的N和D"""
computer_budget_params_loss_list = []
for key in grouped_data:
    grouped_data_min = (grouped_data[key][0]['compute_budget'],grouped_data[key][0]['parameters'],grouped_data[key][0]['final_loss'])
    computer_budget_params_loss_list.append(grouped_data_min)
print(computer_budget_params_loss_list)


"""第三步使用 scipy.optimize.curve_fit 拟合 N_{opt} = \alpha C^a"""
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np
#这里避免直接用指数，最好线转换成log
def power_law(log_C, log_alpha,a):
    return log_alpha + log_C*a

C = [x[0] for x in computer_budget_params_loss_list]
N = [x[1] for x in computer_budget_params_loss_list]
log_C = np.log10(C)
log_N = np.log10(N)

popt, pcov = curve_fit(power_law, log_C, log_N)#这两个参数分别表示拟合的参数和协方差矩阵
print(popt)
x_plot = np.linspace(min(log_C), max(log_C), 1000)
# 绘制“最佳拟合线”
# 使用 *popt 将参数解包传给函数，这是 Pythonic 的写法
y_fit = power_law(x_plot, *popt)
plt.figure(figsize=(10, 6))
# 绘制原始数据散点
plt.scatter(log_C, log_N, label='Raw Data', color='gray', alpha=0.6)
# 绘制最佳拟合线
plt.plot(x_plot, y_fit, 'r-', linewidth=2, label='Best Fit (popt)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.savefig('power_law_fit.png', dpi=300)


"""第四步外推更大的C时的参数量N和D是什么情况"""
C = 1e23
log_C = np.log10(C)
log_N_pred = power_law(log_C, *popt)
N_pred = 10 ** log_N_pred
D_pred = C / (N_pred * 6)
print(f'C={C}, N={N_pred:.2f}, D={D_pred:.2f}')