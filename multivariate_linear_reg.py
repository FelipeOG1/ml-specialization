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
    




"""



import numpy as np
class MultipleLinearRegression:
    def __init__(self):
        self.w = np.array([1.0,2.5,-3.3])
        self.b = 4
        self.x = np.array([10,20,30])


    def sum_no_vectorization(self):return sum([w * x for w,x in zip(self.w,self.x)])





print(MultipleLinearRegression().sum_no_vectorization())
        

        
