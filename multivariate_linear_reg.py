"""
Multifeature Linear Regression


n = num_features
Xj = jth feature
   (i)
Xj     = value of feature j on ith example

->(i)
X     = ith training example (vector of training example)

->(2)
X     = second vector of training example  


Model:  fw,b(x) = w1x1 + ..... wnxn + b
    
Cost Function
  ->
J(w,b)



"""

import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision = 2,suppress = True)
class MultipleLinearRegression:
    def __init__(self):
        self.X = np.array([[2104, 5, 1, 45], 
                           [1416, 3, 2, 40], 
                           [852, 2, 1, 35]],dtype = float)

        self.y_train = np.array([460, 232, 178],dtype = float)
        self.m,self.n = self.X.shape#shape:(rows,cols)
        self.b = 0
        self.w = np.zeros(self.n)
        self.y_hat = lambda x_train:np.dot(x_train,self.w) + self.b 
        self.mean_normalization = lambda train:(train - np.mean(train,axis = 0)) / (np.max(train,axis = 0) - np.min(train,axis = 0))
        self.z_score_normalization = lambda train: (train - np.mean(train,axis = 0)) / (np.std(train,axis = 0))
        self.learning_rate = 0.004
        self.X,self.y_train = self.z_score_normalization(self.X),self.z_score_normalization(self.y_train)
        self.cost_history = []      

    def compute_w_derivate(self):
        hat_minus_train = self.y_hat(self.X) - self.y_train
        return (1 / self.m) * np.dot(self.X.T , (hat_minus_train)) , hat_minus_train #avoid computing y_hat - y_train twice
    def compute_b_derivate(self):return ((self.y_hat(self.X) - self.y_train)).sum() / self.m #this is equivalent to doing np.mean(self.y_hat - self.y_train)
    def compute_cost_function(self):return (1 / (2 * self.m)) * np.sum((self.y_hat(self.X) - self.y_train) ** 2)

    def gradient_descent(self,epsilon = 0.000001,max_iterations = 1000000):
        prev_cost = self.compute_cost_function()
        for i in range(max_iterations):
            w_derivative,hat_minus_train = self.compute_w_derivate()
            self.w = self.w - self.learning_rate * w_derivative
            self.b = self.b - self.learning_rate * np.mean(hat_minus_train)
            current_cost = self.compute_cost_function()
            self.cost_history.append(current_cost)
            print(current_cost,prev_cost)
            if abs(current_cost - prev_cost) < epsilon:
                print(f"convergence on {i}th iteration")
                break    
            prev_cost = current_cost

    def plot_cost_history(self):
        plt.plot(range(len(self.cost_history)),self.cost_history,color = 'blue')
        plt.title("Cost Function vs Iterations")
        plt.xlabel("Iteration")
        plt.ylabel("Cost (J)")
        plt.grid(True)
        plt.savefig("cost_function.png", dpi=300, bbox_inches='tight')

    def __call__(self):
        self.gradient_descent()
        self.plot_cost_history()




MultipleLinearRegression()()





