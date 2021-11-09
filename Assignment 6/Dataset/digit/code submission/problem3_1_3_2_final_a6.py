# Assignment 6, Problem 3.1, 3.2 code written in python
#Name: Mohaiminul Al Nahian
#RIN: 662026703
# Course: CSCI 6100

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(10)

#Data generate

thk=5
rad=10
sep=5

N=1000
sep=5
a_up, b_up=0,sep/2
a_down,b_down=a_up+12.5,-2.5

r_up=np.random.rand(N)*thk+rad
theta_up=np.random.rand(N)*np.pi

x_up=r_up*np.cos(theta_up)+a_up
y_up=r_up*np.sin(theta_up)+b_up

r_up=np.random.rand(N)*thk+rad
theta_up=np.random.rand(N)*np.pi
plt.figure()

plt.scatter(x_up,y_up,color='blue')

x_down=r_up*np.cos(theta_up)+a_up+a_down
y_down=-(r_up*np.sin(theta_up)+sep/2)


plt.scatter(x_down,y_down,color='red')
plt.scatter(a_up,b_up)
plt.scatter(a_down,b_down)
plt.show()

data_up=np.stack([np.ones(N), x_up, y_up], axis=1)
data_down=np.stack([np.ones(N), x_down,y_down],axis=1)
data_all=np.concatenate([data_up, data_down], axis=0)
print(data_up.shape)
y_ground=np.concatenate([np.ones(N), np.ones(N)*-1])



#PLA algorithm

W=np.zeros(3)
iter=0

while 1:
  
  y_pred=(np.dot(data_all,W)>0)*2-1
  
  
  wrong=np.where(y_pred!=y_ground)[0]
   #print (wrong)
  if len(wrong)==0:
    break

  t=wrong[0]
 
  W=W+y_ground[t]*data_all[t]
  iter+=1

print("Total iteration required=")
print(iter)

print('final W value is=   ')
print(W)

print('\n\n')
plt.figure()
color_f='blue'
x=np.linspace(-20,40,100)
g=-(W[1]*x+W[0])/W[2]
color_g='green'
plt.plot(x,g,color=color_g,label='PLA')

plt.scatter(x_up,y_up,color='blue')

plt.scatter(x_down,y_down,color='red')

plt.scatter(a_up,b_up,color='blue')
plt.scatter(a_down,b_down,color='red')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#linear regression

x_reg=np.linalg.inv((np.transpose(data_all)@data_all))@np.transpose(data_all)
print(x_reg.shape)

wlr=np.dot(x_reg,y_ground)
print('final w value for LR\n')
print(wlr)



plt.figure()
color_f='blue'
xlr=np.linspace(-20,40,100)
glr=-(wlr[1]*xlr+wlr[0])/wlr[2]
color_g='green'
plt.plot(x,g,color=color_g,label='PLA')
plt.plot(xlr,glr,color='cyan',label='Linear Regression')

plt.scatter(x_up,y_up,color='blue')

plt.scatter(x_down,y_down,color='red')

plt.scatter(a_up,b_up,color='blue',marker='.')
plt.scatter(a_down,b_down,color='red',marker='.')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

#PROBLEM 3.2


#Data generate

separation=np.arange(0.2,5.01,0.2)
iteration=[]

for i in range(len(separation)):
  np.random.seed(0)
  thk=5
  rad=10
  

  N=1000
  sep=separation[i]
  a_up, b_up=0,sep/2
  a_down,b_down=a_up+12.5,-sep/2

  r_up=np.random.rand(N)*thk+rad
  theta_up=np.random.rand(N)*np.pi

  x_up=r_up*np.cos(theta_up)+a_up
  y_up=r_up*np.sin(theta_up)+b_up

  r_up=np.random.rand(N)*thk+rad
  theta_up=np.random.rand(N)*np.pi


  x_down=r_up*np.cos(theta_up)+a_up+a_down
  y_down=-(r_up*np.sin(theta_up)+sep/2)


  data_up=np.stack([np.ones(N), x_up, y_up], axis=1)
  data_down=np.stack([np.ones(N), x_down,y_down],axis=1)
  data_all=np.concatenate([data_up, data_down], axis=0)
  y_ground=np.concatenate([np.ones(N), np.ones(N)*-1])



  #PLA algorithm

  W=np.zeros(3)
  iter=0

  while 1:
    
    y_pred=(np.dot(data_all,W)>0)*2-1
    
    
    wrong=np.where(y_pred!=y_ground)[0]
    #print (wrong)
    if len(wrong)==0:
      break

    t=wrong[0]
  
    W=W+y_ground[t]*data_all[t]
    iter+=1
  iteration.append(iter)
plt.figure()
plt.plot(separation,iteration)
plt.xlabel('Separation')
plt.ylabel('Iteration')
plt.show()

print (iteration)