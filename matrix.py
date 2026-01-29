
"""

It has super weak rules for matrix validation 
This is just to help me understand the concectps behind  np.dot, matmul,etc.

"""
class Matrix:

    def __init__(self,values:list) -> None:
        if not all(isinstance(value,list) for value in values):
            raise ValueError("invalid matrix")
        if not all(len(values[i]) == len(values[0]) for i in range(1,len(values),1)):
            raise ValueError("Invalid matrix")
    

        self.values = values 

    def __add__(self,other):
       if not isinstance(other,Matrix):
           raise TypeError(f"Cannot sum matrix and {type(other)}")

    


m1 = Matrix([[1,3,3],
             [3,3,3]])
   

