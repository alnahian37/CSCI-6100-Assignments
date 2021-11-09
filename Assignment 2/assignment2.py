import numpy as np
import math
import matplotlib.pyplot as plt

from tqdm import tqdm
# %matplotlib inline

def get_nu():
  headtail=[]
  for i in range(flips):

    headtail.append(np.rint(np.random.rand())) #1 means heads
    
 
  nu=(np.count_nonzero(headtail))/flips 
  return nu

coins=1000
runs=100000
flips=10

nu_1=[]
nu_rand=[]
nu_min=[]
for k in tqdm(range(runs)):
  single_run=[]

  for i in range(coins):
    single_run.append(get_nu())

  nu_1.append(single_run[0])
  nu_rand.append(np.random.choice(single_run))
  nu_min.append(np.min(single_run))

plt.figure(1)
plt.title("Histogram for Nu1")

plt.hist(nu_1,range=(0,1),bins=10)
plt.figure(2)
plt.title("Histogram for Nu_Rand")
plt.hist(nu_rand,range=(0,1),bins=10)
plt.figure(3)
plt.title("Histogram for Nu_min")
plt.hist(nu_min,range=(0,1),bins=10)

eps=np.linspace(0,0.5,100)

val_1=[]
val_rand=[]
val_min=[]

for i in range(len(eps)):
  eps_now=eps[i]
  val_1.append((np.count_nonzero(nu_1>0.5+eps[i])+np.count_nonzero(nu_1<0.5-eps[i]))/len(nu_1))
  val_rand.append((np.count_nonzero(nu_rand>0.5+eps[i])+np.count_nonzero(nu_rand<0.5-eps[i]))/len(nu_1))
  val_min.append((np.count_nonzero(nu_min>0.5+eps[i])+np.count_nonzero(nu_min<0.5-eps[i]))/len(nu_1))


hoeff=2*np.exp(-2*eps**2*flips)

plt.figure(4)

#plt.title("Blue Line=Hoeffding bound, Red line=for minimum Nu, Yellow Line for Nu1, Black dashed line= Random Nu")


plt.plot(eps,val_1,color='yellow',label='Coin1')
plt.plot(eps,val_rand,color='k',linestyle='dashed',label='Random Coin')
plt.plot(eps,val_min,color='r',label='Minimum frequency of heads')
plt.plot(eps,hoeff,color='b', label='Hoeffding bound')
plt.legend()