import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

n_data = torch.ones(100, 2)
x0 = torch.normal(2*n_data, 1)
y0 = torch.zeros(100)
x1 = torch.normal(-2*n_data, 1)
y1 = torch.ones(100)
x = torch.cat((x0, x1), 0).type(torch.FloatTensor)
y = torch.cat((y0, y1), ).type(torch.LongTensor)

x, y = Variable(x), Variable(y)

# plt.scatter(x.data.numpy(), y.data.numpy())
# plt.show()

class Net(torch.nn.Module):
    def __init__(self, n_featur, n_hidden, n_output): #继承层
        super(Net, self).__init__()
        self.hidden=torch.nn.Linear(n_featur, n_hidden) #多少个输入
        self.predict=torch.nn.Linear(n_hidden, n_output) #多少个输入

    def forward(self, x):  #前向传播
        x = F.relu(self.hidden(x))
        x = self.predict(x) #输出层
        return x

net = Net(2, 10, 2) #2个特征，10个神经元，输出2个特征
print(net)

plt.ion() #实时打印
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss() #softmax  概率

for t in range(100):
    out = net(x)

    loss = loss_func(out, y)

    optimizer.zero_grad() #梯度归零
    loss.backward()  #返向传播
    optimizer.step() #

    if t%2 == 0:
        plt.cla()
        prediction = torch.max(F.softmax(out, dim=1), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()
        plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=pred_y, s=100, lw=0, cmap=1)
        accuracy = sum(pred_y == target_y) / 200
        plt.text(1.5, -4, 'Accuracy=%.2f' % accuracy, fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


