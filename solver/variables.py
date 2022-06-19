import numpy as np
import operator as op

from enum import Enum, auto

from expression import Expression


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
        return Expression(self, op.add, other)

    def __sub__(self, other):
        return Expression(self, op.sub, other)

    def __mul__(self, other):
        return Expression(self, op.mul, other)

    def __truediv__(self, other):
        return Expression(self, op.truediv, other)

    def __floordiv__(self, other):
        return Expression(self, op.floordiv, other)

    def __pow__(self, other):
        return Expression(self, op.pow, other)

    def __lt__(self, other):
        return Expression(self, op.lt, other)

    def __le__(self, other):
        return Expression(self, op.le, other)

    def __eq__(self, other):
        return Expression(self, op.eq, other)

    def __ge__(self, other):
        return Expression(self, op.ge, other)
    
    def __gt__(self, other):
        return Expression(self, op.gt, other)

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
