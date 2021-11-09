# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import matplotlib.pyplot as plt


# %%

lr=[0.01,0.1]

for n in range(len(lr)):
    x=0.1
    y=0.1
    fx=[x**2 + 2*y**2 + 2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)]
    xstar=[x]

    ystar=[y]


    for i in range(50):
       
        xnew = x-lr[n]*(2*x + 4*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y))
        ynew = y-lr[n]*(4*y + 4*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y))
        x=xnew
        y=ynew
        f = x**2 + 2*y**2 + 2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
        xstar.append(x)
        ystar.append(y)
        fx.append(f)
    
    print('for Learning rate=   '+str(lr[n]))
    print('minimul loss function value=   '+str(min(fx)))

    print(np.argmin(fx))

    print('point=   '+str(xstar[np.argmin(fx)])+' , '+str(ystar[np.argmin(fx)]))
    plt.figure()
    plt.plot(fx)
    plt.xlabel('iteration')
    plt.ylabel('loss function value')
    plt.show()


# %%
for n in range(len(lr)):
    val=[0.1,1,-0.5,-1]
    for point in range(len(val)):
        x=val[point]

        y=val[point]
        fx=[]
        fx=[x**2 + 2*y**2 + 2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)]
        xstar=[]
        xstar=[x]
        ystar=[]

        ystar=[y]


        for i in range(49):

            
            xnew = x-lr[n]*(2*x + 4*np.pi * np.cos(2*np.pi*x) * np.sin(2*np.pi*y))
            ynew = y-lr[n]*(4*y + 4*np.pi * np.sin(2*np.pi*x) * np.cos(2*np.pi*y))
            x=xnew
            y=ynew
            f = x**2 + 2*y**2 + 2 * np.sin(2*np.pi*x) * np.sin(2*np.pi*y)
            xstar.append(x)
            ystar.append(y)
            fx.append(f)
        
        print('for Learning rate=   '+str(lr[n])+' and initial point=  '+str(val[point])+',' + str(val[point]))
        print('minimul loss function value=   '+str(min(fx)))
        #print(np.argmin(fx))
        print('minimum value point=   '+str(xstar[np.argmin(fx)])+' , '+str(ystar[np.argmin(fx)]))
        #print('\n')
    print('\n')



# %%



# %%
print(len(fx))
print(len(xstar))
print(fx[0])


# %%



