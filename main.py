import torch
import torch.nn as nn
import random
import math
import matplotlib.pyplot as plt

n_in = 2
n_h1 = 6
n_h2 = 6
n_h3 = 6
n_out = 1
batch_size = 10
positive = 100
negative = 100
x = []
y = []
for i in range(positive):
    X = random.randrange(-314, 314, 1) / 100
    Y = math.sin(X) 
    x.append([X, Y])
    y.append([1.0])
for i in range(negative):
    X = random.randrange(-314, 314, 1) / 100
    Y = math.cos(X) 
    x.append([X, Y])
    y.append([0.0])

x = torch.tensor(x)
y = torch.tensor(y)
model = nn.Sequential(
    nn.Linear(n_in, n_h1),
    nn.ReLU(),
    nn.Linear(n_h1, n_h2),
    nn.ReLU(),
    nn.Linear(n_h2, n_h3),
    nn.ReLU(),
    nn.Linear(n_h3, n_out),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(1000):
    y_pred = model(x)#前向传播
    loss = criterion(y_pred, y)#损失计算

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()#更新参数

fig,ax = plt.subplots()
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.grid(True)

for i in range(positive):
    ax.plot(x[i][0], x[i][1], 'ro')  # 使用红色圆点表示正样本
for i in range(positive, positive + negative):
    ax.plot(x[i][0], x[i][1], 'bo')  # 使用蓝色圆点表示负样本
def on_click(event):
    # 检查是否在坐标轴内点击
    if event.inaxes != ax:
        return
    
    # 获取点击的坐标
    x_click, y_click = event.xdata, event.ydata
    x_click = float(x_click)
    y_click = float(y_click)
    # 将点击的坐标添加到列表中
    
    # 在图上绘制点击的点
    new_point = []
    new_point.append([x_click,y_click])
    y = model(torch.tensor(new_point)) # 获取模型预测的概率
    if y.item() > 0.5:
        ax.plot(x_click, y_click, 'ro')  # 使用红色圆点表示预测为正样本
    else:
        ax.plot(x_click, y_click, 'bo')  # 使用蓝色圆点表示预测为负样本
    plt.draw()  # 更新图形显示
cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()