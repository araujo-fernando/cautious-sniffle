import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pprint import pprint
from multiprocessing import freeze_support

from model import Model
from pso import ParticleSwarmOptimizer
from de import DifferentialEvolutionOptimizer
from experiments import realize_experiments

if __name__ == '__main__':
    freeze_support()
    model = Model()
    p = model.create_real_variable("p", 0, 500)
    t = model.create_real_variable("t", 30, 100)
    i = model.create_integer_variable("i", 50, 100)
    obj = 500 * (100 - ((p - 200) ** 2)) * (t ** 0.3) * (i ** 0.25)
    constr = p - 300
    constr2 = t - 60
    constr3 = i - 90
    model.set_objective(obj)
    model.insert_lt_zero_constraint(constr)
    model.insert_lt_zero_constraint(constr2)
    model.insert_eq_zero_constraint(constr3)
    print("Maximize:")
    print(obj)
    print("Subject to:")
    pprint(model._constraints)
    print()

    realize_experiments(model, 500, 500, 300)

quit()




de = DifferentialEvolutionOptimizer(model, max_iterations=100, num_individuals=500)
pso = ParticleSwarmOptimizer(model, max_iterations=300, num_particles=500)
best_individual = de.optimize()
print("DE Solution:")
pprint({name: var._value for name, var in best_individual._vars.items()})
print("With costs:")
pprint(best_individual.objective_values)
print(f"In {de.solve_time} seconds\n")

best_particle = pso.optimize()
print("PSO Solution:")
pprint({name: var._value for name, var in best_particle._vars.items()})
print("With costs:")
pprint(best_particle.objective_values)
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
