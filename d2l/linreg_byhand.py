import torch
from d2l import torch as d2l
import random

def synthetic_data(w,b,num_examples): #@save
# generate data
    x = torch.normal(0,1,(num_examples,len(w)))
    y = torch.matmul(x,w) + b
    y += torch.normal(0,0.01,y.shape)
    return x,y.reshape(-1,1)

def data_iter(batch_size, features, labels):
# batch
    num_examples = len(labels)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0,num_examples, batch_size):
        batch_indices = indices[i:min(i+batch_size,num_examples)]
    yield features[batch_indices], labels[batch_indices]

def linreg(w,b,x): #@save
# line model
    return torch.matmul(x,w)+b

def squared_loss(y_hat,y): #@save
# loss function
    return ((y_hat-y.reshape(y_hat.shape))**2)/2 # why?

def sgd(params, lr, batch_size): #@save
# sgd
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad/batch_size
            param.grad.zero_()   # why no return?


true_w = torch.tensor([2,-3.4])
true_b = 4.2
num_examples = 1000
features, labels = synthetic_data(true_w, true_b, 1000)
batch_size = 10

w = torch.randn((2,1),requires_grad = True)
b = torch.randn(1,requires_grad = True)

# train
lr = 0.1
num_epochs = 200
loss = squared_loss
net = linreg

for epoch in range(num_epochs):
    for x,y in data_iter(batch_size, features, labels):
        loss(net(w,b,x),y).sum().backward()
        sgd([w,b], lr, batch_size)
    with torch.no_grad():
        train_l = loss(net(w,b,features),labels)
        print(f'epoch {epoch+1},loss:{float(train_l.mean()):f}')

print(f'e(w): {true_w - w.reshape(true_w.shape)}')
print(f'e(b): {true_b - b}')
