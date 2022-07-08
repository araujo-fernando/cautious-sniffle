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


def create_resume_table(pso_params: list, de_params: list):
    def extract_data(params: tuple, algo: str="de"):
        iters, indiv, vars, constrs = params
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
            objectives.append(dados["objectives"][0]/1000)
            penalties.append(dados["objectives"][1]/1000)

        return indiv, np.mean(solve_times), np.std(solve_times), np.min(objectives), np.max(objectives), np.mean(objectives), np.std(objectives)

    header = "Variáveis&Restrições&População&Tempo Médio&Pior Objetio&Melhor Objetivo&Média dos Objetivos\n"
    ps_data = ""
    de_data = ""
    comp_data = ""
    for i in range(len(pso_params)):
        pso_param = pso_params[i]
        de_param = de_params[i]
        vars = pso_param[2]
        constrs = pso_param[3]

        ps_pop, ps_time, ps_time_std, ps_min, ps_max, ps_mean, ps_std = extract_data(pso_param, "pso")
        de_pop, de_time, de_time_std, de_min, de_max, de_mean, de_std = extract_data(de_param, "de")
        time_ratio = 100*ps_time/de_time

        ps_data += f"${vars:d}$&${constrs:d}$&"
        ps_data += f"${ps_pop}$&${ps_time:.2f}\\pm{ps_time_std:.2f}$&${ps_min:.2f}$&${ps_max:.2f}$&${ps_mean:.2f}\\pm{ps_std:.2f}$\n"
    
        de_data += f"${vars:d}$&${constrs:d}$&"
        de_data += f"${de_pop}$&${de_time:.2f}\\pm{de_time_std:.2f}$&${de_min:.2f}$&${de_max:.2f}$&${de_mean:.2f}\\pm{de_std:.2f}$\n"

        comp_data += f"${vars:d}$&${constrs:d}$&${ps_pop}$&${time_ratio:.2f}\\%$\n"

    with open("ps_table.txt", "w") as f:
        f.write(header)
        f.write(ps_data)
    with open("de_table.txt", "w") as f:
        f.write(header)
        f.write(de_data)
    with open("comp_table.txt", "w") as f:
        f.write("Variáveis & Restrições & População & Razão de Tempo\n")
        f.write(comp_data)

# ['solve_time', 'evo_data', 'num_vars', 'num_constrs', 'objectives',
#  'solution_variables_values', 'population', 'max_iterations']

path = "solver/results/"
files = os.listdir(path)
pso_files = [f for f in files if f.endswith("pso.json")]
de_files = [f for f in files if f.endswith("de.json")]

regex = r"(\d+)it_(\d+)_ind(\d+)var_(\d+)"

pso_params = list(set(re.match(regex, arq).groups() for arq in pso_files))
de_params = list(set(re.match(regex, arq).groups() for arq in de_files))

# iterations, population, variables, constraints
pso_params = [(int(p[0]), int(p[1]), int(p[2]), int(p[3])) for p in pso_params if p[0] == "1000"]
de_params = [(int(p[0]), int(p[1]), int(p[2]), int(p[3])) for p in de_params if p[0] == "333"]

pso_params = sorted(pso_params, key=lambda p: (p[2], p[3], p[0], p[1]))
de_params = sorted(de_params, key=lambda p: (p[2], p[3], p[0], p[1]))

create_resume_table(pso_params, de_params)

