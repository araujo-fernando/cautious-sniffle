from __future__ import annotations
import os

import random as rd
import json

from concurrent.futures import ProcessPoolExecutor, as_completed
from time import time
from solver import Model, ParticleSwarmOptimizer, DifferentialEvolutionOptimizer


def create_pso(model, num_particles, max_iterations):
    ps = ParticleSwarmOptimizer(
        model, num_particles=num_particles, max_iterations=max_iterations
    )
    return ps

def create_de(model, num_particles, max_iterations):
    de = DifferentialEvolutionOptimizer(
        model, num_individuals=num_particles, max_iterations=max_iterations
    )
    return de

def realize_experiments(
    model: Model,
    PSO_SWARM=500,
    DE_POPULATION=250,
    PSO_ITERATIONS=900,
):
    DE_ITERATIONS = PSO_ITERATIONS // 3

    with ProcessPoolExecutor(10) as exec:
        futures = [
            exec.submit(
                create_pso,
                model.copy(),
                PSO_SWARM,
                PSO_ITERATIONS,
            )
            for _ in range(10)
        ]
        for future in as_completed(futures):
            pso = future.result()
            dump_json_results(pso)

    with ProcessPoolExecutor(10) as exec:
        futures = [
            exec.submit(
                create_de,
                model.copy(),
                DE_POPULATION,
                DE_ITERATIONS,
            )
            for _ in range(10)
        ]
        for future in as_completed(futures):
            pso = future.result()
            dump_json_results(pso)


def dump_json_results(
    optimizer: ParticleSwarmOptimizer | DifferentialEvolutionOptimizer,
):
    solve_time = optimizer.solve_time
    solution = optimizer.solution
    evo_data = optimizer.evolution_data
    num_vars = solution.num_vars
    num_constrs = len(solution._constraints)
    objectives = solution.objective_values
    solution_variables_values = solution.variables_values

    if isinstance(optimizer, ParticleSwarmOptimizer):
        file_name = "pso_"
        max_iterations = optimizer.max_iterations
        population = optimizer.num_particles
    else:
        file_name = "de_"
        max_iterations = optimizer.max_iterations
        population = optimizer.num_individuals
    file_name += f"{max_iterations}it_{population}_ind{num_vars}var_{num_constrs}cnstr"

    experiment_data = {
        "solve_time": solve_time,
        "evo_data": evo_data,
        "num_vars": num_vars,
        "num_constrs": num_constrs,
        "objectives": objectives,
        "solution_variables_values": solution_variables_values,
        "population": population,
        "max_iterations": max_iterations,
    }
    to_dump = json.dumps(experiment_data)

    for i in range(10):
        name = file_name + f"_{i}.json"
        if not os.path.isfile(name):
            file_name = name
            break

    with open(file_name + ".json", "w") as f:
        f.write(to_dump)


def assemble_model(TOTAL_NOS=10, T=60):
    TOTAL_MERCADORIAS = TOTAL_NOS // 2
    model = Model()

    ## GERAÇÃO DOS DADOS BASE
    start_time = time()
    mercadorias = [f"mercadoria_{x}" for x in range(TOTAL_MERCADORIAS)]
    nos = [f"no_{x}" for x in range(TOTAL_NOS)]
    nos_clientes = rd.sample(nos, TOTAL_NOS // 5)
    nos_fornecedores = list(
        rd.sample(list(set(nos) - set(nos_clientes)), TOTAL_NOS // 5)
    )
    nos_intermediarios = list(set(nos) - set(nos_clientes) - set(nos_fornecedores))
    todos_pares = [
        (i, j) for i in nos for j in nos if (i != j) and (i not in nos_clientes)
    ]
    todos_pares_mercadorias = [(i, j, m) for i, j in todos_pares for m in mercadorias]
    fornecedores_mercadorias = [(i, m) for i in nos_fornecedores for m in mercadorias]
    ## GERAÇÃO DAS VARIÁVEIS
    p_0_m = model.create_real_variables("p_0_", mercadorias, lb=10, ub=500)
    p_1_m = model.create_real_variables("p_1_", mercadorias, lb=500, ub=1000)
    p_2_m = model.create_real_variables("p_2_", mercadorias, lb=250, ub=750)

    s_0_i_m = model.create_integer_variables(
        "s_0_", nos_intermediarios, lb=3000, ub=4000
    )
    s_1_i_m = model.create_integer_variables(
        "s_1_", nos_intermediarios, lb=2000, ub=3000
    )
    s_2_i_m = model.create_integer_variables(
        "s_2_", nos_intermediarios, lb=1000, ub=2000
    )

    b_i = model.create_binary_variables("b_", nos)

    c_i_j_m = model.create_real_variables("c_", todos_pares_mercadorias, lb=0, ub=100)
    f_i_j_m = model.create_real_variables("f_", todos_pares_mercadorias, lb=0, ub=100)
    g_j_m = model.create_integer_variables(
        "g_", fornecedores_mercadorias, lb=50, ub=150
    )
    w = model.create_real_variable("w", lb=0, ub=T)

    ## GERAÇÃO DAS CONSTANTES
    d_j_m = {(i, m): rd.uniform(10, 100) for i in nos_clientes for m in mercadorias}
    h_i_m = {(i, m): rd.uniform(100, 200) for i in nos for m in mercadorias}
    e_i = {i: rd.uniform(1000, 10000) for i in nos if i not in nos_clientes}
    h_m = {m: rd.uniform(0, 20) for m in mercadorias}
    v_i = {i: rd.uniform(50, 500) for i in nos}
    u_i = {i: rd.uniform(5, 50) for i in nos}

    alpha_m = {m: rd.uniform(0.001, 0.999) for m in mercadorias}
    eps_m = {m: rd.uniform(0.001, 0.999) for m in mercadorias}
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
            sum(
                (c_i_j_m[(i, j, m)] + v_i[i]) * f_i_j_m[(i, j, m)]
                for i, j in todos_pares
            )
            for m in mercadorias
        )
        - sum(u_i[i] * b_i[i] for i in nos)
    )

    model.set_objective(objetivo)
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
    r_20 = [
        g_j_m.get((i, m), 0) - h_i_m.get((i, m), 0) for i in nos for m in mercadorias
    ]
    r_21 = [
        d_j_m.get((j, m), 0)
        - sum(f_i_j_m[(i, j, m)] for i in nos if (i, j) in todos_pares)
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
        s_2_i_m.get((i, m), 0) - s_1_i_m.get((i, m), 0)
        for i in nos
        for m in mercadorias
    ]
    r_26_2 = [
        s_1_i_m.get((i, m), 0) - s_0_i_m.get((i, m), 0)
        for i in nos
        for m in mercadorias
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

    return model, (end_time - start_time)
