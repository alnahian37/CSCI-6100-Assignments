

# Assignment 1, Problem 1.4(1-e) code written in python
#Name: Mohaiminul Al Nahian
#RIN: 662026703
# Course: CSCI 6100

import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline

np.random.seed(0)
n=20 #Dataset length for 1.5 (a,b)

x1=np.random.rand(n)*2-1

x2=np.random.rand(n)

x_f=np.linspace(-1,1,20)
m=-0.7
c=.5
y_f=m*x_f+c #Target funtion

X=np.transpose([np.ones(n), x1, x2])


index=np.zeros(n)

for i in range(n):
  if x2[i]>m*x1[i]+c:
    index[i]=1
posindex=np.where(index==1)[0]
negindex=np.where(index==0)[0]


y_ground=np.zeros(n)
y_ground[posindex]=1
y_ground[negindex]=-1


plt.figure(1)

color_f='blue'
plt.plot(x_f,y_f,color=color_f)
plt.scatter(x1[posindex],x2[posindex],color='cyan')
plt.scatter(x1[negindex],x2[negindex],color='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(color_f+ ' Line=f (target funtion)')

#PLA algorithm

W=np.zeros(3)
iter=0

while 1:
  
  y_pred=(np.dot(W,np.transpose(X))>0)*2-1
  
  
  wrong=np.where(y_pred!=y_ground)[0]
   #print (wrong)
  if len(wrong)==0:
    break

  t=wrong[0]
 
  W=W+y_ground[t]*X[t]
  iter+=1

print("Total iteration required=")
print(iter)

print('final W value is=   ')
print(W)

print('\n\n')
plt.figure(2)
color_f='blue'
plt.plot(x_f,y_f,color=color_f)
plt.scatter(x1[posindex],x2[posindex],color='cyan')
plt.scatter(x1[negindex],x2[negindex],color='r')
g=-(W[1]*x_f+W[0])/W[2]
color_g='green'
plt.plot(x_f,g,color=color_g)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(color_f+ ' Line=f (target funtion) and '+color_g+' Line=g')

np.random.seed(10)
n=20 #Dataset length for 1.5 (c)

x1=np.random.rand(n)*2-1

x2=np.random.rand(n)

x_f=np.linspace(-1,1,20)
m=-0.7
c=.5
y_f=m*x_f+c #Target funtion

X=np.transpose([np.ones(n), x1, x2])


index=np.zeros(n)

for i in range(n):
  if x2[i]>m*x1[i]+c:
    index[i]=1
posindex=np.where(index==1)[0]
negindex=np.where(index==0)[0]


y_ground=np.zeros(n)
y_ground[posindex]=1
y_ground[negindex]=-1


plt.figure(3)

color_f='blue'
plt.plot(x_f,y_f,color=color_f)
plt.scatter(x1[posindex],x2[posindex],color='cyan')
plt.scatter(x1[negindex],x2[negindex],color='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(color_f+ ' Line=f (target funtion)')

#PLA algorithm

W=np.zeros(3)
iter=0

while 1:
  
  y_pred=(np.dot(W,np.transpose(X))>0)*2-1
  
  
  wrong=np.where(y_pred!=y_ground)[0]
   #print (wrong)
  if len(wrong)==0:
    break

  t=wrong[0]
 
  W=W+y_ground[t]*X[t]
  iter+=1

print("Total iteration required=")
print(iter)

print('final W value is=   ')
print(W)

print('\n\n')
plt.figure(4)
color_f='blue'
plt.plot(x_f,y_f,color=color_f)
plt.scatter(x1[posindex],x2[posindex],color='cyan')
plt.scatter(x1[negindex],x2[negindex],color='r')
g=-(W[1]*x_f+W[0])/W[2]
color_g='green'
plt.plot(x_f,g,color=color_g)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(color_f+ ' Line=f (target funtion) and '+color_g+' Line=g')

np.random.seed(0)
n=100 #Dataset length for 1.5 (d)

x1=np.random.rand(n)*2-1

x2=np.random.rand(n)

x_f=np.linspace(-1,1,20)
m=-0.7
c=.5
y_f=m*x_f+c #Target funtion

X=np.transpose([np.ones(n), x1, x2])


index=np.zeros(n)

for i in range(n):
  if x2[i]>m*x1[i]+c:
    index[i]=1
posindex=np.where(index==1)[0]
negindex=np.where(index==0)[0]


y_ground=np.zeros(n)
y_ground[posindex]=1
y_ground[negindex]=-1


plt.figure(5)

color_f='blue'
plt.plot(x_f,y_f,color=color_f)
plt.scatter(x1[posindex],x2[posindex],color='cyan')
plt.scatter(x1[negindex],x2[negindex],color='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(color_f+ ' Line=f (target funtion)')

#PLA algorithm

W=np.zeros(3)
iter=0

while 1:
  
  y_pred=(np.dot(W,np.transpose(X))>0)*2-1
  
  
  wrong=np.where(y_pred!=y_ground)[0]
   #print (wrong)
  if len(wrong)==0:
    break

  t=wrong[0]
 
  W=W+y_ground[t]*X[t]
  iter+=1

print("Total iteration required=")
print(iter)

print('final W value is=   ')
print(W)

print('\n\n')
plt.figure(6)
color_f='blue'
plt.plot(x_f,y_f,color=color_f)
plt.scatter(x1[posindex],x2[posindex],color='cyan')
plt.scatter(x1[negindex],x2[negindex],color='r')
g=-(W[1]*x_f+W[0])/W[2]
color_g='green'
plt.plot(x_f,g,color=color_g)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(color_f+ ' Line=f (target funtion) and '+color_g+' Line=g')

np.random.seed(0)
n=1000 #Dataset length for 1.5 (e)

x1=np.random.rand(n)*2-1

x2=np.random.rand(n)

x_f=np.linspace(-1,1,20)
m=-0.7
c=.5
y_f=m*x_f+c #Target funtion

X=np.transpose([np.ones(n), x1, x2])


index=np.zeros(n)

for i in range(n):
  if x2[i]>m*x1[i]+c:
    index[i]=1
posindex=np.where(index==1)[0]
negindex=np.where(index==0)[0]


y_ground=np.zeros(n)
y_ground[posindex]=1
y_ground[negindex]=-1


plt.figure(7)

color_f='blue'
plt.plot(x_f,y_f,color=color_f)
plt.scatter(x1[posindex],x2[posindex],color='cyan')
plt.scatter(x1[negindex],x2[negindex],color='r')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(color_f+ ' Line=f (target funtion)')

#PLA algorithm

W=np.zeros(3)
iter=0

while 1:
  
  y_pred=(np.dot(W,np.transpose(X))>0)*2-1
  
  
  wrong=np.where(y_pred!=y_ground)[0]
   #print (wrong)
  if len(wrong)==0:
    break

  t=wrong[0]
 
  W=W+y_ground[t]*X[t]
  iter+=1

print("Total iteration required=")
print(iter)

print('final W value is=   ')
print(W)

print('\n\n')
plt.figure(8)
color_f='blue'
plt.plot(x_f,y_f,color=color_f)
plt.scatter(x1[posindex],x2[posindex],color='cyan')
plt.scatter(x1[negindex],x2[negindex],color='r')
g=-(W[1]*x_f+W[0])/W[2]
color_g='green'
plt.plot(x_f,g,color=color_g)
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(color_f+ ' Line=f (target funtion) and '+color_g+' Line=g')