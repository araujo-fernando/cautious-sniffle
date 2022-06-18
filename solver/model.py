

class Model:
    def __init__(self) -> None:
        self.obj = list()
        self.constraints = set()
    
    def get_objective_x(self, id: int):
        if id >= len(self.obj):
            raise ValueError(f"ID must be between 0 and {len(self.obj)-1}.")
        
        return self.obj[id]

    def set_objective_x(self, expression, id: int=0):
        if id > len(self.obj):
            raise ValueError(f"ID must be between 0 and {len(self.obj)}.")

        self.obj[id] = expression

    def set_objective(self, expression):
        self.set_objective_x(expression, 0)
