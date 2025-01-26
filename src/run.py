import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import sys
sys.path.append("/Users/sigur.de.vries/Library/Mobile Documents/com~apple~CloudDocs/phd/MultiTreeGP")

import jax
import jax.numpy as jnp
import jax.random as jrandom

import MultiTreeGP.evaluators.dynamic_evaluate as dynamic_evaluate
import MultiTreeGP.evaluators.static_evaluate as static_evaluate
import CMAES_evaluate as CMAES_evaluate
import LQG_evaluate as LQG_evaluate
from NDE import ParameterReshaper

from MultiTreeGP.genetic_programming import GeneticProgramming
from CMAES import CMA_ES

from MultiTreeGP.environments.control_environments.harmonic_oscillator import HarmonicOscillator, ChangingHarmonicOscillator
from MultiTreeGP.environments.control_environments.reactor import StirredTankReactor
from MultiTreeGP.environments.control_environments.acrobot import Acrobot, Acrobot2
import diffrax

def get_data(key, env, batch_size, dt, T, param_setting):
    init_key, noise_key1, noise_key2, param_key = jrandom.split(key, 4)
    x0, targets = env.sample_init_states(batch_size, init_key)
    process_noise_keys = jrandom.split(noise_key1, batch_size)
    obs_noise_keys = jrandom.split(noise_key2, batch_size)
    ts = jnp.arange(0, T, dt)

    params = env.sample_params(batch_size, param_setting, ts, param_key)
    return x0, ts, targets, process_noise_keys, obs_noise_keys, params


def run(env_string, algorithm_string, seed = 0, param_setting = "Constant"):
    key = jrandom.PRNGKey(seed)
    init_key, data_key = jrandom.split(key)

    population_size = 100
    num_populations = 10
    num_generations = 50
    state_size = 2
    T = 50
    dt = 0.2
    batch_size = 8

    process_noise = 0.05
    obs_noise = 0.1

    if env_string=="HO":
        if param_setting == "Changing":
            env = ChangingHarmonicOscillator(process_noise, obs_noise, n_obs=2)
        else:
            env = HarmonicOscillator(process_noise, obs_noise, n_obs=2)

        operator_list = [("+", lambda x, y: x + y, 2, 0.5), 
                    ("-", lambda x, y: x - y, 2, 0.1),
                    ("*", lambda x, y: x * y, 2, 0.5),
                    ("/", lambda x, y: x / y, 2, 0.1),
                    ("**", lambda x, y: x ** y, 2, 0.1),
                    ]

    elif env_string=="ACR":
        env = Acrobot(process_noise, obs_noise)

        # env = Acrobot2(process_noise, obs_noise, n_obs=None)

        operator_list = [("+", lambda x, y: x + y, 2, 0.5), 
                        ("-", lambda x, y: x - y, 2, 0.1),
                        ("*", lambda x, y: x * y, 2, 0.5),
                        ("/", lambda x, y: x / y, 2, 0.1),
                        ("**", lambda x, y: x ** y, 2, 0.1),
                        ("sin", lambda x: jnp.sin(x), 1, 0.1),
                        ("cos", lambda x: jnp.cos(x), 1, 0.1)
                        ]

    elif env_string=="CSTR":
        env = StirredTankReactor(process_noise, obs_noise, n_obs=2)

        operator_list = [("+", lambda x, y: x + y, 2, 0.5), 
                        ("-", lambda x, y: x - y, 2, 0.1),
                        ("*", lambda x, y: x * y, 2, 0.5),
                        ("/", lambda x, y: x / y, 2, 0.1),
                        ("**", lambda x, y: x ** y, 2, 0.1),
                        ("exp", lambda x: jnp.exp(x), 1, 0.1),
                        ("log", lambda x: jnp.log(x), 1, 0.1)
                        ]

    data = get_data(data_key, env, batch_size, dt, T, param_setting)

    if algorithm_string == "Static":
        fitness_function = static_evaluate.Evaluator(env, 0.01, max_steps = 10000, solver=diffrax.GeneralShARK())

        variable_list = [["y" + str(i) for i in range(env.n_obs)] + ["tar" + str(i) for i in range(env.n_targets)]]

        layer_sizes = jnp.array([env.n_control])

        strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes,
                                num_populations = num_populations)

    elif algorithm_string == "Dynamic" or algorithm_string == "Random":
        fitness_function = dynamic_evaluate.Evaluator(env, state_size, 0.01, max_steps = 10000, solver=diffrax.GeneralShARK())

        variable_list = [["y" + str(i) for i in range(env.n_obs)] + ["a" + str(i) for i in range(state_size)] + ["u"] + ["tar" + str(i) for i in range(env.n_targets)],
                         ["a" + str(i) for i in range(state_size)]]#  + ["tar" + str(i) for i in range(env.n_targets)]]

        layer_sizes = jnp.array([state_size, env.n_control])
        
        strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes, 
                                num_populations = num_populations)
        
    elif algorithm_string == "NDE":
        latent_dim = 5
        parameter_reshaper = ParameterReshaper(env.n_obs + env.n_control, latent_dim, env.n_control, env.n_targets)
        fitness_function = CMAES_evaluate.Evaluator(env, parameter_reshaper, latent_dim, 0.01, max_steps = 10000, solver=diffrax.GeneralShARK())

        key, cma_key = jrandom.split(key)
        strategy = CMA_ES(num_generations, population_size*num_populations, fitness_function, parameter_reshaper.total_parameters, cma_key)

    elif algorithm_string == "LQG":
        fitness_function = LQG_evaluate.Evaluator(env, 0.01)
        _,_,_, fitness = fitness_function(data)

        print(fitness)

        return fitness

    population = strategy.initialize_population(init_key)

    for g in range(num_generations):
        key, eval_key, sample_key = jrandom.split(key, 3)
        fitness, population = strategy.evaluate_population(population, data, eval_key)
        
        if algorithm_string in ["Static", "Dynamic", "Random"]:
            best_fitness, best_solution = strategy.get_statistics(g)
            print(f"In generation {g+1}, best fitness = {best_fitness:.4f}, best solution = {strategy.to_string(best_solution)}")
        else:
            best_fitness = strategy.get_statistics(g)
            print(f"In generation {g+1}, best fitness = {best_fitness:.4f}")

        if g < (num_generations-1):
            
            if algorithm_string == "Random":
                population = strategy.initialize_population(sample_key)
                population[0,0] = best_solution
                strategy.increase_generation()
            else:
                population = strategy.evolve(population, fitness, sample_key)

    return strategy.get_statistics()

param_settings = ["Constant", "Different", "Changing"]
envs = ["HO", "ACR", "CSTR"]
algorithms = ["Static", "Dynamic", "Random", "NDE", "LQG"]
seed = 1

# best_fitness, best_solutions = run("HO", "Static", seed, "Constant")
best_fitness, best_solutions = run("ACR", "Dynamic", seed, "Constant")
# best_fitness, best_solutions = run("HO", "Random", seed, "Constant")
# best_fitness = run("ACR", "NDE", seed, "Constant")
# fitness = run("HO", "LQG", seed, "Constant")
