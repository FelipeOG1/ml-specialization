
class Matrix:
    def __init__(self, values: list) -> None:
        if not all(isinstance(value, list) for value in values):
            raise ValueError("invalid matrix: rows must be lists")
        if not all(len(row) == len(values[0]) for row in values):
            raise ValueError("invalid matrix: all rows must have the same length")
        
        self.values = values


    @property
    def shape(self):return len(self.values),len(self.values[0])
    
    @property
    def T(self):
        rows,cols = self.shape
        return Matrix([[row[i] for row in self.values] for i in range(cols)])

    def column(self,col):return [row[col] for row in self.values]
    
            
    def __repr__(self) -> str:
        return f"Matrix({self.values})"

    def __iter__(self):
        return iter(self.values)


    def __getitem__(self,index):
        if isinstance(index,tuple):
            row,col = index
            return self.values[row][col]
        
        return self.values[index]
        
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

    

    def dot_scalar(self,m2):
        new_matrix = self * m2
        _sum = 0
        for row in new_matrix:
            _sum+=sum(row)

        return _sum

    def dot_matrix(self,m2):
        m1_rows,m1_cols = self.shape
        m2_rows,m2_cols = m2.shape

        valid = m1_cols == m2_rows
        if not valid:
            raise ValueError("Cannot operate dot")

        new_values = []
        for i in range(m1_rows):
            row = self.values[i]
            new_row = []
            for j in range(m2_cols):
                
                col = m2.column(j)
                new_scalar = sum([r * c for r,c in zip(row,col)])
                new_row.append(new_scalar)

                
            new_values.append(new_row)


        return Matrix(new_values)

