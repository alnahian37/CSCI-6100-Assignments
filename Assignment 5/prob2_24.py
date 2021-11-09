import numpy as np
import math
import matplotlib.pyplot as plt

x=np.linspace(-1,1,1000)
len(x)
fx=x**2
#plt.plot(x,fx)

M=[]
C=[]
Eout=[]
np.random.seed(100)
for i in range(1000):
  x1=np.random.rand()*2-1
  x2=np.random.rand()*2-1
  M.append(x1+x2)
  C.append(x1*x2)
  gx = (x1+x2) *x - x1*x2
  e_out = np.mean(np.square( fx- gx))
  Eout.append(e_out)

E_out=np.mean(Eout)


mbar=np.mean(M)
cbar=np.mean(C)
gbarx=mbar*x+cbar

bias=np.mean(np.square(gbarx-fx))
plt.plot(x,gbarx,label='g_bar(x)')
plt.plot(x,fx,label='f(x)')
plt.legend()

var = []
for i in range(len(M)):
    
    gx = M[i]*x + C[i]
    #gbar = a_avg * X_test + b_avg
    va = np.mean(np.square(gx - gbarx))
    var.append(va)
var_out = np.mean(var)
print('E[Eout]=',E_out)
print('variance=',var_out)
print('bias=',bias)
print('bias+variance=',var_out+bias)
print('mbar,cbar=',mbar,cbar)