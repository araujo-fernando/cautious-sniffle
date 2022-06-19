

class Expression:
    def __init__(self, a, op, b) -> None:
        self.a = a
        self.op = op
        self.b = b

    @property
    def value(self):
        return self.op(self.a.value, self.b.value)
            