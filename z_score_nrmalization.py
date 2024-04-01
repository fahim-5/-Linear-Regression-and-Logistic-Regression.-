import numpy as np
import copy, math

def zscore_normalize_features(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    X_norm = (X - mu) / sigma
    return X_norm, mu, sigma

def compute_cost(X, y, w, b): 
    m = X.shape[0]
    cost = 0.0
    for i in range(m):                                
        f_wb_i = np.dot(X[i], w) + b           
        cost = cost + (f_wb_i - y[i])**2       
    cost = cost / (2 * m)                          
    return cost

def compute_gradient(X, y, w, b): 
    m,n = X.shape           
    dj_dw = np.zeros((n,))
    dj_db = 0.
    for i in range(m):                             
        err = (np.dot(X[i], w) + b) - y[i]   
        for j in range(n):                         
            dj_dw[j] = dj_dw[j] + err * X[i, j]    
        dj_db = dj_db + err                        
    dj_dw = dj_dw / m                                
    dj_db = dj_db / m                                
    return dj_db, dj_dw

def gradient_descent(X, y, w_in, b_in, cost_function, gradient_function, alpha, num_iters): 
    J_history = []
    w = np.zeros(X.shape[1])  
    b = b_in
    for i in range(num_iters):
        dj_db,dj_dw = gradient_function(X, y, w, b)   
        w = w - alpha * dj_dw               
        b = b - alpha * dj_db              
        if i<100000:     
            J_history.append( cost_function(X, y, w, b))
        if i% math.ceil(num_iters / 100) == 0:
            print(f"Iteration {i:4d}: Cost {J_history[-1]:8.2f}   ")
    return w, b, J_history 

def get_user_input():
    features = []
    print("Enter the features of the house:")
    size = float(input("Size in square feet: "))
    bedrooms = int(input("Number of bedrooms: "))
    floors = int(input("Number of floors: "))
    age = int(input("Age of the house: "))
    features.extend([size, bedrooms, floors, age])
    return np.array(features)

X_train = np.array([[2104, 5, 1, 45], [1416, 3, 2, 40], [852, 2, 1, 35]])
y_train = np.array([460, 232, 178])

# Normalize the features
X_train_norm, mu, sigma = zscore_normalize_features(X_train)

n = X_train_norm.shape[0]
initial_w = np.zeros(n)
initial_b = 0.0
iterations = 10000
alpha = 0.0000007

# Perform gradient descent on normalized features
w_final, b_final, J_hist = gradient_descent(X_train_norm, y_train, initial_w, initial_b, compute_cost, compute_gradient, alpha, iterations)

print(f"b,w found by gradient descent: {b_final:0.2f},{w_final} ")
m,n = X_train_norm.shape
for i in range(m):   
    print(f"prediction: {np.dot(X_train_norm[i], w_final) + b_final:0.2f}, target value: {y_train[i]}")

x_values = get_user_input()

# Normalize the user input features using the same mu and sigma from training data
x_values_norm = (x_values - mu) / sigma

result = np.dot(x_values_norm, w_final) + b_final
print("Estimated price of the house:", result)
