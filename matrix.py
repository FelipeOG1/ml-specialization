


class Matrix:
    def __init__(self,values:list) -> None:
        if not all(isinstance(value,list) for value in values):
            raise ValueError("invalid matrix")


        self.values = values 
        
    
    @property
    def shape(self):
        base_len = len(self.values[0])
        if any(len(value) != base_len for value in self.values[1:]):
            raise ValueError("invalid sequence")


        
        
    
        
    






