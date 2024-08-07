import torch
import torch.nn as nn
import numpy as np

class net(nn.Module):
    def __init__(self, in_size: int, out_size: int):
        super(net,self).__init__()
        self.lin1 = nn.Linear(in_size, out_features=5)
        self.lin2 = nn.Linear(in_features=5, out_features=out_size)
    
    def forward(self,x):
        return(self.lin2(self.lin1(x)))
    
model = net(in_size = 10, out_size=2)
x = torch.rand(size=(1,10))
print(model.forward(x))

# a = torch.ones((2,2), dtype = torch.float32)
# b = torch.zeros((2,2))

# c = torch.ones_like(a, dtype=torch.float32)


# # print(c.type())
# # print(c.ndimension(), c.size())
# # print(c.shape)

# d = [[1,5], [22,-1]]
# d = torch.tensor(d)

# # print(d, d.max(), d.min())
# # print(d.tolist())
# # print(d.numpy())
# # e = np.array(d)
# # print(e, type())

# #addition and broadcasting

# a = [[1,2], [3,4]]
# a = torch.tensor(a, dtype = torch.float)
# # b =[3]
# # b = torch.tensor(b)
# # print(a+b)

# #changing dim 

# # a = a.view(1,4)
# # print(a)

# # print(a.mean())

# print(torch.sin(a))

# a = torch.linspace(-10, 10, 20)
# print(a.view(5,4))

# a = np.linspace(-5,5, num = 10)
# a = torch.tensor(a)
# print(torch.norm(a))
# # print(a.view(5,2))

# ##derivative in torch
# x = torch.tensor(2.0, requires_grad=True)
# y = x**2
# y.backward()
# print(x.grad)

# ##partial derivate in torch

# u = torch.tensor(1.0, requires_grad=True)
# v = torch.tensor(2.0, requires_grad=True)

# f = (u*v)+(u**2)

# f.backward()
# print(u.grad, v.grad)

class net(nn.Mdule):
    def __init__(self, in_ch ,out_ch):
        super(net,self).__init__()
        self.l1 = nn.Linear(in_ch, out_ch)

    def forward(self,x):
        return self.l1(x)