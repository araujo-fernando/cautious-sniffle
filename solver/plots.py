from __future__ import annotations

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import json
import os
import re

def plot_evolution_heat_map(evolution_data: list[list[list[float]]]):
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


def load_file_data(file_name: str):
    with open(file_name, "r") as f:
        data = json.loads(f.read())

    return data

def create_resume_table(params: list, algo: str = "pso"):
    print("\n", algo+":")
    for iters, indiv, vars, constrs in params:
        experiment_files = [
            path + f"{iters}it_{indiv}_ind{vars}var_{constrs}cnstr_{i}_{algo}.json"
            for i in range(10)
        ]

        solve_times = list()
        objectives = list()
        penalties = list()
        for experiment in experiment_files:
            dados = load_file_data(experiment)
            solve_times.append(dados["solve_time"]/60)
            objectives.append(dados["objectives"][0])
            penalties.append(dados["objectives"][1])

        line = f"{indiv:4d}\t{vars:4d}\t{constrs:4d}\t{np.mean(solve_times):5.2f}±{np.std(solve_times):3.2f}\t{np.mean(objectives):5.2f}±{np.std(objectives):6.2f}"
        print(line)


# ['solve_time', 'evo_data', 'num_vars', 'num_constrs', 'objectives',
#  'solution_variables_values', 'population', 'max_iterations']

path = "solver/results/"
files = os.listdir(path)
pso_files = [f for f in files if f.endswith("pso.json")]
de_files = [f for f in files if f.endswith("de.json")]

regex = r"(\d+)it_(\d+)_ind(\d+)var_(\d+)"

pso_params = list(set(re.match(regex, arq).groups() for arq in pso_files))
de_params = list(set(re.match(regex, arq).groups() for arq in de_files))

pso_params = [(int(p[0]), int(p[1]), int(p[2]), int(p[3])) for p in pso_params if p[0] == "1000"]
de_params = [(int(p[0]), int(p[1]), int(p[2]), int(p[3])) for p in de_params if p[0] == "333"]

pso_params = sorted(pso_params, key=lambda p: (p[2], p[3], p[0], p[1]))
de_params = sorted(de_params, key=lambda p: (p[2], p[3], p[0], p[1]))

create_resume_table(pso_params, "pso")
create_resume_table(de_params, "de")

