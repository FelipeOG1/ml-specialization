
"""

It has super weak rules for matrix validation 
This is just to help me understand the concectps behind  np.dot, matmul,etc.

"""
class Matrix:
    def __init__(self, values: list) -> None:
        if not all(isinstance(value, list) for value in values):
            raise ValueError("invalid matrix: rows must be lists")
        if not all(len(row) == len(values[0]) for row in values):
            raise ValueError("invalid matrix: all rows must have the same length")
        
        self.values = values


    @property
    def shape(self):return len(self.values),len(self.values[0])
    
    
                   
            
    def __repr__(self) -> str:
        return f"Matrix({self.values})"

    def __iter__(self):
        return iter(self.values)

    def __len__(self):
        return len(self.values)
    
    def __add__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot sum matrix and {type(other)}")
        
        if len(self) != len(other) or len(self.values[0]) != len(other.values[0]):
            raise ValueError("Matrices must have the same dimensions to be added")

        return Matrix(
            [
            [a + b for a, b in zip(row_self, row_other)]
            for row_self, row_other in zip(self, other)
            ]
        
        )

    def __mul__(self, other):
        if not isinstance(other, Matrix):
            raise TypeError(f"Cannot sum matrix and {type(other)}")
        
        if len(self) != len(other) or len(self.values[0]) != len(other.values[0]):
            raise ValueError("Matrices must have the same dimensions to be added")

        return Matrix(
            [
            [a * b for a, b in zip(row_self, row_other)]
            for row_self, row_other in zip(self, other)
            ]
        
        )

    def dot(self,m2):
        new_matrix = self * m2
        _sum = 0
        for row in new_matrix:
            _sum+=sum(row)

        return _sum
        

    
          
