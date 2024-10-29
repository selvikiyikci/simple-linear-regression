## Simple Regression
#     Make sure you add the bias feature to each training and test example.
#     Standardize the features using the mean and std computed over training data.

import sys
import numpy as np
from matplotlib import pyplot as plt
import scaling


# Read data matrix X and labels y from text file.
def read_data(file_name):
#  YOUR CODE HERE
        data = np.genfromtxt(file_name, delimiter=' ', dtype=float) 
        data = data[~np.isnan(data).any(axis=1)]

        if data.size == 0:
            print(f"Veri dosyası '{file_name}' boş veya geçersiz!")
            sys.exit(1)

        X = data[:, 0].reshape(-1, 1)
        y = data[:, 1]

        if len(X) == 0 or len(y) == 0:
            print(f"Veri dosyası '{file_name}' boş veya geçersiz!")
            sys.exit(1)

        return X, y

  
# Implement gradient descent algorithm to compute w = [w0, w1].
def train(X, y, lamda, epochs):
#  YOUR CODE HERE
    w = np.zeros(X.shape[1]) 
    J_history = [] 

    for epoch in range(epochs):
        grad = compute_gradient(X, y, w) 
        w -= lamda * grad 
        cost = compute_cost(X, y, w)  
        J_history.append(cost) 

    return w, J_history 


# Compute Root mean squared error (RMSE)).
def compute_rmse(X, y, w):
#  YOUR CODE HERE
    predictions = X.dot(w)
    rmse = np.sqrt(np.mean(np.square(predictions - y)))  
    return rmse


# Compute objective (cost) function.
def compute_cost(X, y, w):
#  YOUR CODE HERE
    m = len(y)  
    predictions = X.dot(w)  
    cost = (1 / (2 * m)) * np.sum(np.square(predictions - y))  # Maliyet hesaplaması
    return cost

# Compute gradient descent Algorithm.
def compute_gradient(X, y, w):
#  YOUR CODE HERE
    m = len(y)  
    predictions = X.dot(w) 
    grad = (1 / m) * (X.T.dot(predictions - y)) 
    return grad



##======================= Main program =======================##


Xtrain, ttrain = read_data("train.txt")
Xtest, ttest = read_data("test.txt")

#  YOUR CODE HERE

Xtrain = np.c_[np.ones(Xtrain.shape[0]), Xtrain]  
Xtest = np.c_[np.ones(Xtest.shape[0]), Xtest]    

mean = np.mean(Xtrain[:, 1:], axis=0)  
std = np.std(Xtrain[:, 1:], axis=0)   
Xtrain[:, 1:] = scaling.standardize(Xtrain[:, 1:], mean, std)  
Xtest[:, 1:] = scaling.standardize(Xtest[:, 1:], mean, std)    

lamda = 0.1
epochs = 500
w, J_history = train(Xtrain, ttrain, lamda, epochs)

print("Ağırlıklar:", w)
rmse_train = compute_rmse(Xtrain, ttrain, w)
rmse_test = compute_rmse(Xtest, ttest, w)
print("RMSE (Train seti):", rmse_train)
print("RMSE (Test seti):", rmse_test)

# Cost
plt.plot(J_history)
plt.xlabel("Epoch")
plt.ylabel("Cost J(w)")
plt.title("J(w) vs. Epochs")
plt.show()


plt.scatter(Xtrain[:, 1], ttrain, color='blue', label='Train Verisi') 
plt.scatter(Xtest[:, 1], ttest, color='green', label='Test Verisi') 

# Linear solution line
x_values = np.linspace(Xtrain[:, 1].min(), Xtrain[:, 1].max(), 100)
y_values = w[0] + w[1] * ((x_values - mean) / std) 
plt.plot(x_values, y_values, color='red', label='Linear Solution')
plt.xlabel('Floor Size')
plt.ylabel('Cost')
plt.title('Price Prediction Based on Floor Size')
plt.legend()
plt.show()