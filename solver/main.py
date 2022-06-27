import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint

from solver import *

TOTAL_MERCADORIAS = 100
TOTAL_NOS = 10

model = Model()

mercadorias = [f"mercadoria_{x}" for x in range(TOTAL_MERCADORIAS)]
nos = [f"no_{x}" for x in range(TOTAL_NOS)]
todos_pares = [(i, j) for i in nos for j in nos if i != j]
todos_pares_mercadorias = [(i, j, m) for i in nos for j in nos if i != j for m in mercadorias]

p_0_m = model.create_real_variables("p_0_", mercadorias, lb=0, ub=1000)
p_1_m = model.create_real_variables("p_1_", mercadorias, lb=0, ub=1000)
p_2_m = model.create_real_variables("p_2_", mercadorias, lb=0, ub=1000)

s_0_m = model.create_integer_variables("s_0_", mercadorias, lb=0, ub=5000)
s_1_m = model.create_integer_variables("s_0_", mercadorias, lb=0, ub=5000)
s_2_m = model.create_integer_variables("s_0_", mercadorias, lb=0, ub=5000)

vi = model.create_real_variables("c_", nos, lb=0, ub=500)
ui = model.create_real_variables("u_", nos, lb=0, ub=500)

c_i_j_m = model.create_real_variables("c_", todos_pares_mercadorias, lb=0, ub=100)
f_i_j_m = model.create_real_variables("f_", todos_pares_mercadorias, lb=0, ub=100)


print("Model Statistics:")
print(f"{len(model._vars)} variables")
print(f"{len(model._constraints)} constraints\n")

de = DifferentialEvolutionOptimizer(model, max_iterations=100, num_individuals=500)
pso = ParticleSwarmOptimizer(model, max_iterations=300, num_particles=500)
print("DE Solution:")
de_solution = de.optimize()
pprint({name: var._value for name, var in de_solution._vars.items()})
print("With costs:")
pprint(de_solution.objective_values)
print(f"In {de.solve_time} seconds\n")

print("PSO Solution:")
pso_solution = pso.optimize()
pprint({name: var._value for name, var in pso_solution._vars.items()})
print("With costs:")
pprint(pso_solution.objective_values)
print(f"In {pso.solve_time} seconds\n")

## DE METRICS
evolution_data = de.evolution_data
objectives = [[val[0] for val in it] for it in evolution_data]
penalties = [[val[1] for val in it] for it in evolution_data]

for data in objectives:
    data.sort()
for data in penalties:
    data.sort()

objectives = list(map(list, zip(*objectives)))
penalties = list(map(list, zip(*penalties)))

objectives = np.array(objectives)
min_value = objectives.min()
max_value = objectives.max()
objectives = (objectives - min_value) / (max_value - min_value)

penalties = np.array(penalties)
min_value = penalties.min()
max_value = penalties.max()
penalties = (penalties - min_value) / (max_value - min_value)

objs_evo = sns.heatmap(objectives)
plt.show()

pen_evo = sns.heatmap(penalties)
plt.show()

## PSO METRICS
evolution_data = pso.evolution_data
objectives = [[val[0] for val in it] for it in evolution_data]
penalties = [[val[1] for val in it] for it in evolution_data]

for data in objectives:
    data.sort()
for data in penalties:
    data.sort()

objectives = list(map(list, zip(*objectives)))
penalties = list(map(list, zip(*penalties)))

objectives = np.array(objectives)
min_value = objectives.min()
max_value = objectives.max()
objectives = (objectives - min_value) / (max_value - min_value)

penalties = np.array(penalties)
min_value = penalties.min()
max_value = penalties.max()
penalties = (penalties - min_value) / (max_value - min_value)

objs_evo = sns.heatmap(objectives)
plt.show()

pen_evo = sns.heatmap(penalties)
plt.show()
