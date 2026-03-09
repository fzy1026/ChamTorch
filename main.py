import torch
import torch.nn as nn

n_in = 2
n_h1 = 4
n_h2 = 4
n_out = 1
batch_size = 10
positive = 0
negative = 0

x = []
y = []
with open('data.txt', 'r') as f:
    i = 1
    for line in f:
        if(i == 1):
            positive = float(line.strip())
        elif(i == positive + 2):
            negative = float(line.strip())
        else:
            line = line.strip().split()
            x.append([float(line[0]), float(line[1])])
            if i <= positive:
                y.append([1])
            else:
                y.append([0])
        i += 1
x = torch.tensor(x)
y = torch.tensor(y)
model = nn.Sequential(
    nn.Linear(n_in, n_h1),
    nn.ReLU(),
    nn.Linear(n_h1, n_h2),
    nn.ReLU(),
    nn.Linear(n_h2, n_out),
    nn.Sigmoid()
)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
for i in range(1000):
    y_pred = model(x)#前向传播
    loss = criterion(y_pred, y.float())#损失计算

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()#更新参数

while 1:
    testX = input("请输入测试数据（格式：x1 x2）：")
    testX = torch.tensor([list(map(float, testX.split()))])
    y_test = model(testX)
    if y_test.item() > 0.5:
        print("预测结果：正样本")
    else:
        print("预测结果：负样本")