import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.set_title("Click on the plot")
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.grid(True)

def on_click(event):
    # 检查是否在坐标轴内点击
    if event.inaxes != ax:
        return
    
    # 获取点击的坐标
    x, y = event.xdata, event.ydata
    
    # 在图上绘制点击的点
    ax.plot(x, y, 'ro')  # 使用红色圆点表示点击的位置
    plt.draw()  # 更新图形显示

cid = fig.canvas.mpl_connect('button_press_event', on_click)
plt.show() 