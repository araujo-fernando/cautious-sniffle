import numpy as np


from enum import Enum, auto


class VarType(Enum):
    BINARY = auto()
    INTEGER = auto()
    REAL = auto()


class _Variable:
    def __init__(self, name, lb=None, ub=None) -> None:
        self.name = name
        self.lb = lb
        self.ub = ub

        self.value = None
        self.type = None

    def set_value(self, v):
        self._value = np.clip(v, self.lb, self.ub)

    def __setattr__(self, name, value):
        if name == "value":
            self.set_value(value)
        self.__dict__[name] = value

    def __add__(self, other):
        return self.value + other.value

    def __iadd__(self, other):
        self.set_value(self.value + other)
        return self

    def __sub__(self, other):
        return self.value - other.value

    def __isub__(self, other) :
        self.set_value(self.value - other)
        return self
    
    def __mul__(self, other):
        return self.value * other.value
    
    def __imul__(self, other) :
        self.set_value(self.value * other)
        return self
    
    def __truediv__(self, other):
        return self.value / other.value
    
    def __itruediv__(self, other) :
        self.set_value(self.value + other.value)
        return self
    
    def __floordiv__(self, other) :
        return self.value // other.value


class BinVariable(_Variable):
    def __init__(self, name) -> None:
        super().__init__(name, 0, 1)
        self.type = VarType.BINARY

    def set_value(self, v):
        self._value = int(v >= 0.5)


class IntVariable(_Variable):
    def __init__(self, name, lb=None, ub=None) -> None:
        super().__init__(name, lb, ub)
        self.type = VarType.INTEGER

    def set_value(self, v):
        self._value = np.round(np.clip(v, self.lb, self.ub))


class RealVariable(_Variable):
    def __init__(self, name, lb=None, ub=None) -> None:
        super().__init__(name, lb, ub)
        self.type = VarType.REAL

    def set_value(self, v):
        self._value = np.clip(v, self.lb, self.ub)

    
