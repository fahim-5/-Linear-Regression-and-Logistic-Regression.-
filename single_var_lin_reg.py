import numpy as np

def gradient_decent(x,y) :
    w=0
    b=0
    iteration=10000
    learning_rate=0.02
    n=len(x)
    for i in range(iteration):
        y_predicted=w*x+b     # this y_predicted is important to  deteermine the value of dw and bd
        wd=(1/n)*sum((y_predicted-y)*x)
        bd=(1/n)*sum(y_predicted-y)
        
        w=w-learning_rate*wd
        b=b-learning_rate*bd

        cost = (1/n) * sum((y_predicted - y)**2)

        if i % 100 == 0:            
            print(f"Iteration {i}: w = {w}, b = {b}, cost = {cost}")
    return w,b    
    
x=np.array([1,2,3,4,5])
y=np.array([5,7,9,11,13])


m,b=gradient_decent(x,y)

x=int(input("Enter a number : " ))  
result= liner_refrassion(x)
print(f"your ans is : {result} ") 

