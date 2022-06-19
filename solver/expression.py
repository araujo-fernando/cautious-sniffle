
_OPERADORES = {
    "add": "+",
    "sub": "-",
    "mul": "*",
    "truediv": "/",
    "floordiv": "//",
    "pow": "^",
    "lt": "<",
    "le": "<=",
    "eq": "=",
    "ge": ">=",
    "gt": ">",
}

class Expression:
    def __init__(self, a, op, b) -> None:
        self.a = a
        self.op = op
        self.b = b

    @property
    def value(self):
        return self.op(self.a.value, self.b.value)

    def __repr__(self) -> str:
        op_name = self.op.__name__
        op = _OPERADORES.get(op_name, op_name)
        return f"{self.a} {op} {self.b}"
            
    def __str__(self) -> str:
        op_name = self.op.__name__
        op = _OPERADORES.get(op_name, op_name)
        return f"({self.a} {op} {self.b})"
