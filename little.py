
from turtle import shape
import torch

            # Bs,Cs,Hs,Ws= x_.shape

x=torch.rand(4, 256,56,340)
Bs,Cs,Hs,Ws= x.shape
x1= x[:,:,0:40:,0:23]
print(x.shape)
print(x1.shape)