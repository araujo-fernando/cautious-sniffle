from __future__ import annotations
from copy import deepcopy

from expression import Expression
from variables import RealVariable, BinVariable, IntVariable


class Model:
    def __init__(self) -> None:
        self._objectives: list[Expression] = list()
        self._vars: dict[str, RealVariable | BinVariable | IntVariable] = dict()

    @property
    def num_vars(self):
        return len(self._vars)

    @property
    def variables(self):
        return deepcopy(self._vars)

    @property
    def objectives(self):
        return deepcopy(self._objectives)

    def get_objective_x(self, id: int):
        if id >= len(self._objectives):
            raise ValueError(f"ID must be between 0 and {len(self._objectives)-1}.")

        return self._objectives[id]

    def set_objective_x(self, expression, id: int = 0):
        if id > len(self._objectives):
            raise ValueError(f"ID must be between 0 and {len(self._objectives)}.")
        
        if id == len(self._objectives):
            self._objectives.append(expression)
        else:
            self._objectives[id] = expression

    def set_objective(self, expression):
        self.set_objective_x(expression, 0)

    def create_binary_variables(self, data: list):
        new_vars = {val: self._vars.get(val, BinVariable(val)) for val in data}
        for k, v in new_vars.items():
            self._vars[k] = v
        return new_vars

    def create_integer_variables(self, data: list, lb=None, ub=None):
        new_vars = {val: self._vars.get(val, IntVariable(val, lb, ub)) for val in data}
        for k, v in new_vars.items():
            self._vars[k] = v
        return new_vars

    def create_real_variables(self, data: list, lb=None, ub=None):
        new_vars = {val: self._vars.get(val, RealVariable(val, lb, ub)) for val in data}
        for k, v in new_vars.items():
            self._vars[k] = v
        return new_vars

    def create_binary_variable(self, name: str):
        new_var = self._vars.get(name, BinVariable(name))
        self._vars[name] = new_var
        return new_var

    def create_integer_variable(self, name: str, lb=None, ub=None):
        new_var = self._vars.get(name, IntVariable(name, lb, ub))
        self._vars[name] = new_var
        return new_var

    def create_real_variable(self, name: str, lb=None, ub=None):
        new_var = self._vars.get(name, RealVariable(name, lb, ub))
        self._vars[name] = new_var
        return new_var

    def copy(self) -> Model:
        return deepcopy(self)

    def set_variables_values(self, var_values: dict[str, int | float]):
        for var, val in var_values.items():
            if var in self._vars:
                self._vars[var].value = val
