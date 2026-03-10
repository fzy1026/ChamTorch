import torch
import torch.nn as nn
import random
import math
import matplotlib.pyplot as plt

num = 100
catagory = 5
colors = [
    'red',
    'blue',
    'green',
    'yellow',
    'black'
]
center = [
    [5, -7.5],
    [5, 5],
    [-5, 5],
    [0,0],
    [-5, -5]
]

n_in = 2
n_h1 = 6
n_h2 = 6
n_h3 = 6
n_out = catagory
batch_size = 10
x = []
y = []

def generate_data():
    for i in range(catagory):
        for j in range(num):
            r = random.uniform(0, 3)
            angle = random.uniform(0, 2*math.pi)
            R = random.uniform(0, r)
            x.append([center[i][0] + R*math.cos(angle), center[i][1] + R*math.sin(angle)])
            Y = [0.0] * catagory
            Y[i] = 1.0
            y.append(Y)




generate_data()
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
print(x)


fig,ax = plt.subplots()
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)
ax.grid(True)
for i in range(num * catagory):
    ax.plot(x[i][0], x[i][1], colors[torch.argmax(y[i])], marker='o')  # 使用不同颜色表示不同类别的点

    

def on_click(event):
    """
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
    ax.plot(x_click, y_click, colors[torch.argmax(y)], marker='o')  # 使用不同颜色表示不同类别的点
    plt.draw()  # 更新图形显示
    """
    i = -10.0
    while (i<10.0):
        j = -10.0
        while (j<10.0):
            new_point = []
            new_point.append([i,j])
            y = model(torch.tensor(new_point)) # 获取模型预测的概率
            ax.plot(i, j, colors[torch.argmax(y)], marker='o')  # 使用不同颜色表示不同类别的点
            j += 0.8
        i += 0.8
    plt.draw()  # 更新图形显示

    
    

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show()