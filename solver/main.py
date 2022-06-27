import numpy as np
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint
from time import time

from solver import *

## CONFIGURAÇÕES PARA SOLVERS
PSO_SWARM = 500
DE_POPULATION = 500

PSO_ITERATIONS = 300
DE_ITERATIONS = 100
## CONFIGURAÇÕES PARA MONTAGEM DO PROBLEMA
T = 60
TOTAL_NOS = 10
TOTAL_MERCADORIAS = 2 * TOTAL_NOS

model = Model()

## GERAÇÃO DOS DADOS BASE
start_time = time()
mercadorias = [f"mercadoria_{x}" for x in range(TOTAL_MERCADORIAS)]
nos = [f"no_{x}" for x in range(TOTAL_NOS)]
nos_clientes = rd.sample(nos, TOTAL_NOS // 5)
nos_fornecedores = list(rd.sample(list(set(nos) - set(nos_clientes)), TOTAL_NOS // 5))
nos_intermediarios = list(set(nos) - set(nos_clientes) - set(nos_fornecedores))
todos_pares = [(i, j) for i in nos for j in nos if (i != j) and (i not in nos_clientes)]
todos_pares_mercadorias = [(i, j, m) for i, j in todos_pares for m in mercadorias]
fornecedores_mercadorias = [(i, m) for i in nos_fornecedores for m in mercadorias]
## GERAÇÃO DAS VARIÁVEIS
p_0_m = model.create_real_variables("p_0_", mercadorias, lb=0, ub=1000)
p_1_m = model.create_real_variables("p_1_", mercadorias, lb=0, ub=1000)
p_2_m = model.create_real_variables("p_2_", mercadorias, lb=0, ub=1000)

s_0_i_m = model.create_integer_variables("s_0_", nos_intermediarios, lb=0, ub=5000)
s_1_i_m = model.create_integer_variables("s_1_", nos_intermediarios, lb=0, ub=5000)
s_2_i_m = model.create_integer_variables("s_2_", nos_intermediarios, lb=0, ub=5000)

v_i = model.create_real_variables("c_", nos, lb=0, ub=500)
u_i = model.create_real_variables("u_", nos, lb=0, ub=500)
b_i = model.create_binary_variables("b_", nos)

c_i_j_m = model.create_real_variables("c_", todos_pares_mercadorias, lb=0, ub=100)
f_i_j_m = model.create_real_variables("f_", todos_pares_mercadorias, lb=0, ub=100)
g_j_m = model.create_integer_variables("g_", fornecedores_mercadorias, lb=50, ub=150)
w = model.create_real_variable("w", lb=0, ub=T)

## GERAÇÃO DAS CONSTANTES
d_j_m = {(i, m): rd.uniform(10, 100) for i in nos_clientes for m in mercadorias}
h_i_m = {(i, m): rd.uniform(100, 200) for i in nos for m in mercadorias}
e_i = {i: rd.uniform(1000, 10000) for i in nos if i not in nos_clientes}
h_m = {m: rd.uniform(0, 20) for m in mercadorias}

alpha_m = {m: rd.uniform(0.001, 2) for m in mercadorias}
eps_m = {m: rd.uniform(0.001, 2) for m in mercadorias}
gama_m = {m: rd.uniform(0.001, 2) for m in mercadorias}
beta_m = {m: rd.uniform(0.001, 0.999) for m in mercadorias}
r_m = {m: alpha_m[m] * beta_m[m] / gama_m[m] for m in mercadorias}

## GERAÇÃO DA FUNÇÃO OBJETIVO
objetivo = (
    sum(
        sum(
            (
                (p_1_m[m] - p_0_m[m]) * s_0_i_m.get((i, m), 0)
                - (p_1_m[m] - p_2_m[m]) * s_1_i_m.get((i, m), 0)
                - (p_2_m[m] + h_m[m]) * s_2_i_m.get((i, m), 0)
            )
            for m in mercadorias
        )
        for i in nos_clientes
    )
    - sum(
        sum((c_i_j_m[(i, j, m)] + v_i[i]) * f_i_j_m[(i, j, m)] for i, j in todos_pares)
        for m in mercadorias
    )
    - sum(u_i[i] * b_i[i] for i in nos)
)

## GERAÇÃO DAS RESTRIÇÕES DE IGUALDADE
r_18 = [
    g_j_m.get((j, m), 0)
    + s_0_i_m.get((j, m), 0)
    + sum(f_i_j_m[(i, j, m)] for i in nos if (i, j) in todos_pares)
    - d_j_m.get((j, m), 0)
    + s_2_i_m.get((j, m), 0)
    - sum(f_i_j_m[(j, k, m)] for k in nos if (j, k) in todos_pares)
    for j in nos
    for m in mercadorias
]
model.insert_eq_zero_constraints(r_18)

## GERAÇÃO DAS RESTRIÇÕES DE MENOR IGUAL
r_19 = [
    sum(f_i_j_m[(i, j, m)] for j in nos if (i, j) in todos_pares) - e_i.get(i, 0)
    for i in nos
    for m in mercadorias
]
r_20 = [g_j_m.get((i, m), 0) - h_i_m.get((i, m), 0) for i in nos for m in mercadorias]
r_21 = [
    d_j_m.get((j, m), 0) - sum(f_i_j_m[(i, j, m)] for i in nos if (i, j) in todos_pares)
    for j in nos_clientes
    for m in mercadorias
]
r_22 = [
    (
        s_0_i_m.get((i, m), 0) ** beta_m[m]
        - r_m[m] * (w ** gama_m[m]) / (p_1_m[m] ** eps_m[m])
    )
    ** (1 / beta_m[m])
    - s_1_i_m.get((i, m), 0)
    for i in nos
    for m in mercadorias
]
r_23 = [
    (
        s_0_i_m.get((i, m), 0) ** beta_m[m]
        - r_m[m] * (w ** gama_m[m]) / (p_1_m[m] ** eps_m[m])
        - r_m[m] * (T ** gama_m[m] - w ** gama_m[m]) / (p_2_m[m] ** eps_m[m])
    )
    ** (1 / beta_m[m])
    - s_2_i_m.get((i, m), 0)
    for i in nos
    for m in mercadorias
]
r_24_1 = [p_0_m[m] - p_2_m[m] for m in mercadorias]
r_24_2 = [p_2_m[m] - p_1_m[m] for m in mercadorias]
r_26_1 = [
    s_2_i_m.get((i, m), 0) - s_1_i_m.get((i, m), 0) for i in nos for m in mercadorias
]
r_26_2 = [
    s_1_i_m.get((i, m), 0) - s_0_i_m.get((i, m), 0) for i in nos for m in mercadorias
]

model.insert_lt_zero_constraints(r_19)
model.insert_lt_zero_constraints(r_20)
model.insert_lt_zero_constraints(r_21)
model.insert_lt_zero_constraints(r_22)
model.insert_lt_zero_constraints(r_23)
model.insert_lt_zero_constraints(r_24_1)
model.insert_lt_zero_constraints(r_24_2)
model.insert_lt_zero_constraints(r_26_1)
model.insert_lt_zero_constraints(r_26_2)
end_time = time()
print("Model Statistics:")
print(f"{len(model._vars)} variables")
print(f"{len(model._constraints)} constraints\n")
print(f"Created in {end_time-start_time} seconds\n")

de = DifferentialEvolutionOptimizer(
    model, max_iterations=DE_ITERATIONS, num_individuals=DE_POPULATION
)
pso = ParticleSwarmOptimizer(
    model, max_iterations=PSO_ITERATIONS, num_particles=PSO_SWARM
)
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
