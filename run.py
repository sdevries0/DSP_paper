import os
os.environ["XLA_FLAGS"] = '--xla_force_host_platform_device_count=10'

import sys
sys.path.append("/Users/sigur.de.vries/Library/Mobile Documents/com~apple~CloudDocs/phd/MultiTreeGP")

import jax
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib.pyplot as plt
import sympy
import diffrax


import MultiTreeGP.evaluators.dynamic_evaluate as dynamic_evaluate
import MultiTreeGP.evaluators.feedforward_evaluate as ff_evaluate
import MultiTreeGP.evaluators.lqg_evaluate as lqg_evaluate

from MultiTreeGP.genetic_programming import GeneticProgramming

from MultiTreeGP.environments.control_environments.harmonic_oscillator import HarmonicOscillator, HarmonicOscillator2, ChangingHarmonicOscillator
from MultiTreeGP.environments.control_environments.reactor import StirredTankReactor
from MultiTreeGP.environments.control_environments.cart_pole import CartPole
from MultiTreeGP.environments.control_environments.acrobot import Acrobot, Acrobot2

def my_to_string(sol):
    expr = sympy.parsing.sympy_parser.parse_expr(sol,evaluate=True)
    return simplification.sympy_to_tree(expr,mode="Add")

def get_data(key, env, batch_size, dt, T, param_setting):
    init_key, noise_key1, noise_key2, param_key = jrandom.split(key, 4)
    x0, targets = env.sample_init_states(batch_size, init_key)
    process_noise_keys = jrandom.split(noise_key1, batch_size)
    obs_noise_keys = jrandom.split(noise_key2, batch_size)
    ts = jnp.arange(0, T, dt)

    params = env.sample_params(batch_size, param_setting, ts, param_key)
    return x0, ts, targets, process_noise_keys, obs_noise_keys, params


key = jrandom.PRNGKey(0)
init_key, data_key = jrandom.split(key)

population_size = 100
num_populations = 3
num_generations = 50
state_size = 2
T = 50
dt = 0.2
batch_size = 8

process_noise = 0.05
obs_noise = 0.1

param_setting = "Constant"


def run(env_string, algorithm_string):

    if env_string=="HO":
        env = HarmonicOscillator(process_noise, obs_noise, n_obs=2)
        # env = ChangingHarmonicOscillator(process_noise, obs_noise, n_obs=2)

    elif env_string=="ACR":
        env = Acrobot(process_noise, obs_noise)

        # env = Acrobot2(process_noise, obs_noise, n_obs=None)

    elif env_string=="CSTR":
        env = StirredTankReactor(process_noise, obs_noise, n_obs=2)

    operator_list = [("+", lambda x, y: x + y, 2, 0.5), 
                    ("-", lambda x, y: x - y, 2, 0.1),
                    ("*", lambda x, y: x * y, 2, 0.5),
                    # ("/", lambda x, y: x / y, 2, 0.1),
                #  ("**", lambda x, y: x ** y, 2, 0.1),
                ("sin", lambda x: jnp.sin(x), 1, 0.1),
                ("cos", lambda x: jnp.cos(x), 1, 0.1)
                # ("exp", lambda x: jnp.exp(x), 1, 0.1),
                #  ("log", lambda x: jnp.log(x), 1, 0.1)
                    ]

    data = get_data(data_key, env, batch_size, dt, T, param_setting)

    if algorithm_string == "FF":
        fitness_function = ff_evaluate.Evaluator(env, 0.01, stepsize_controller=diffrax.PIDController(atol=1e-6, rtol=1e-6, dtmin=0.001), max_steps=500)

        variable_list = [["y" + str(i) for i in range(env.n_obs)], ["a" + str(i) for i in range(state_size)] + ["u"] + ["tar" + str(i) for i in range(env.n_targets)]]

        layer_sizes = jnp.array([env.n_control])

        strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes,
                                num_populations = num_populations, coefficient_sd=3, max_depth=8, max_nodes=25, max_init_depth=4)

    elif algorithm_string == "GP":
        fitness_function = dynamic_evaluate.Evaluator(env, state_size, 0.05, solver=diffrax.Dopri5(), stepsize_controller=diffrax.PIDController(atol=1e-4, rtol=1e-4, dtmin=0.001), max_steps=1000)

        variable_list = [["y" + str(i) for i in range(env.n_obs)] + ["a" + str(i) for i in range(state_size)] + ["u"]]

        layer_sizes = jnp.array([state_size, env.n_control])
        
        strategy = GeneticProgramming(num_generations, population_size, fitness_function, operator_list, variable_list, layer_sizes, 
                                num_populations = num_populations, coefficient_sd=1, max_init_depth=4, max_nodes=25, coefficient_optimisation=True, gradient_steps=15)

    elif algorithm_string == "LQG":
        fitness_function = lqg_evaluate.Evaluator(env)
        _,_,_, fitness = fitness_function(data)

    population = strategy.initialize_population(init_key)

    for g in range(num_generations):
        fitness, population = strategy.evaluate_population(population, data)
        best_fitness, best_solution = strategy.get_statistics(g)
        print(f"In generation {g+1}, best fitness = {best_fitness:.4f}, best solution = {strategy.to_string(best_solution)}")

        if g < (num_generations-1):
            key, sample_key = jrandom.split(key)
            population = strategy.evolve(population, fitness, sample_key)

    best_fitnesses, best_solutions = strategy.get_statistics()