import torch
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)
y = x.pow(2) + 0.2*torch.rand(x.size())

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

net = Net(1, 10, 1)
print(net)

plt.ion() #实时打印
plt.show()

optimizer = torch.optim.SGD(net.parameters(), lr=0.2)
loss_func = torch.nn.MSELoss() #均方差

for t in range(100):
    prediction = net(x)

    loss = loss_func(prediction, y)

    optimizer.zero_grad() #梯度归零
    loss.backward()  #返向传播
    optimizer.step() #

    if t%5 == 0:
        plt.cla()
        plt.scatter(x.data.numpy(), y.data.numpy())
        plt.plot(x.data.numpy(), prediction.data.numpy(), 'r-', lw=5)
        plt.text(0.5, 0, 'loss=%.4f' % loss.data, fontdict={'size':20, 'color':'red'})
        plt.pause(0.1)

plt.ioff()
plt.show()


