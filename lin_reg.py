import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#accessing data from csv file

data=pd.read_csv('D:\Python_D\linear_regression.csv')
data.head()
arr= np.array(data.values,'float')
arr[0:5,:]
#extract input and output variable from the matrix of data
X= arr[:,0]
Y= arr[:,1]

#feature scaling

X=X/(np.max(X))
plt.plot(X,Y,"bo")
plt.ylabel('Y')
plt.xlabel('X')
plt.title('Data')
plt.grid()
plt.show()

#hypothesis is taken as H(x)= aX+b
#getting the cost function

def costfunc(x,y,theta):
    temp= theta.T*x
    temp=temp[:,0]+temp[:,1]
    sq=np.sum((temp- y)**2)
    j=(1/(2*m))*sq
    return j
m=np.size(Y)
X=X.reshape([m,1])
#Y=Y.reshape([m,1])
x = np.hstack([np.ones_like(X),X])

#initialising theta as 0 vector
theta=np.zeros([2,1])
print(theta,'\n',m)
print(costfunc(x,Y,theta))
#gradient descent
def gradient(x,y,theta):
    alpha=0.0000001
    iters=20000
    J=np.zeros([iters,1])
    for i in range(0,iters):
        temp= theta.T*x
        temp=temp[:,0]+temp[:,1]
        delta=(temp- y)
        temp0=theta[0]-((alpha/m)*np.sum(delta*x[:,0]))
        temp1=theta[1]-((alpha/m)*np.sum(delta*x[:,1]))
        theta = np.array([temp0,temp1]).reshape([2,1])
        J[i] = (1 / (2*m) ) * (np.sum((delta)**2))
    return theta, J
#get optimal theta now

theta, J=gradient(x,Y,theta)
#print(theta)

plt.plot(X,Y,'bo')
temp= theta.T*x
temp=temp[:,0]+temp[:,1]
plt.plot(X,temp,'-')
plt.grid()
plt.show()

#predicting for new input value
inp=int(input("Enter input variable X"))
out= theta[0] + theta[1]*inp
print(out)