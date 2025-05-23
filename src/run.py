import os
import sys

print("nr of cpus", os.cpu_count())
import jax
print(jax.devices())

import jax.numpy as jnp
import jax.random as jrandom

from miscellaneous.expression import Expression

import evaluators.dynamic_evaluate as dynamic_evaluate
import evaluators.cma_es_evaluator as cma_evaluate
import evaluators.static_evaluate as ff_evaluate
import evaluators.lqg_evaluate as LQG_evaluate

from algorithms.genetic_programming import GeneticProgramming
from algorithms.cma_es import CMA_ES
from algorithms.random_search import RandomSearch

from environments.harmonic_oscillator import HarmonicOscillator
from environments.reactor import StirredTankReactor
from environments.acrobot import Acrobot

def get_data(key, env, batch_size, dt, T, param_setting):
    """
    Get the data for the environment
    """
    init_key, noise_key1, noise_key2, param_key = jrandom.split(key, 4)
    x0, targets = env.sample_init_states(batch_size, init_key)
    process_noise_keys = jrandom.split(noise_key1, batch_size)
    obs_noise_keys = jrandom.split(noise_key2, batch_size)
    ts = jnp.arange(0,T,dt)

    params = env.sample_params(batch_size, param_setting, ts, param_key)
    return x0, ts, targets, process_noise_keys, obs_noise_keys, params

def run(seed, program, args):
    key = jrandom.PRNGKey(seed)
    init_key, data_key = jrandom.split(key)

    env_string, param_setting, n_obs, num_generations, num_populations, process_noise, obs_noise = args

    # Define the parameters of the algorithm
    population_size = 100

    state_size = 2
    T = 30
    dt = 0.2
    pool_size = os.cpu_count()
    batch_size = 32

    # Define the environment
    if env_string=="HO":
        env = HarmonicOscillator(process_noise, obs_noise, n_obs=n_obs)

        operators_prob = jnp.array([0.5, 0.3, 0.5, 0.1, 0.1])
        unary_functions = []
        dt0 = 0.02

    elif env_string=="ACR":
        env = Acrobot(process_noise, obs_noise, n_obs=n_obs)
        
        operators_prob = jnp.array([0.5, 0.3, 0.5, 0.1, 0.1, 0.1, 0.1])
        unary_functions = ["sin", "cos"]
        dt0 = 0.02

    elif env_string=="CSTR":
        env = StirredTankReactor(process_noise, obs_noise, n_obs=n_obs)

        operators_prob = jnp.array([0.5, 0.3, 0.5, 0.1, 0.1, 0.1, 0.1])
        unary_functions = ["exp", "log"]
        dt0 = 0.0002

    # Get the data
    data = get_data(data_key, env, batch_size, dt, T, param_setting)

    # Define the fitness function and algorithm
    if program == "Static":
        fitness_function = ff_evaluate.Evaluator(env, state_size, dt0)

        # Define the expressions for the layers
        layer_expressions = [Expression(obs_size=env.n_obs, target_size=env.n_targets, unary_functions = unary_functions, operators_prob=operators_prob)]

        layer_sizes = jnp.array([env.n_control])
        assert len(layer_sizes) == len(layer_expressions), "There is not a set of expressions for every type of layer"

        strategy = GeneticProgramming(num_generations, population_size, fitness_function, layer_expressions, layer_sizes, 
                                num_populations = num_populations, state_size = state_size, pool_size = pool_size)
    
    elif program == "Random":
        fitness_function = dynamic_evaluate.Evaluator(env, state_size, dt0)

        # Define the expressions for the layers
        layer_expressions = [Expression(obs_size=env.n_obs, state_size=state_size, control_size=env.n_control, target_size=env.n_targets, unary_functions = unary_functions, operators_prob=operators_prob), 
                            Expression(obs_size=0, state_size=state_size, control_size=0, target_size=env.n_targets, unary_functions = unary_functions,
                                condition=lambda self, tree: sum([leaf in self.state_variables for leaf in jax.tree_util.tree_leaves(tree)])==0, operators_prob = operators_prob)]

        layer_sizes = jnp.array([state_size, env.n_control])
        assert len(layer_sizes) == len(layer_expressions), "There is not a set of expressions for every type of layer"

        strategy = RandomSearch(num_generations, population_size, fitness_function, layer_expressions, layer_sizes, num_populations = num_populations, state_size = state_size, pool_size = pool_size)
    
    elif program == "NDE":
        fitness_function = cma_evaluate.Evaluator(env, state_size, dt0)
        strategy = CMA_ES(num_generations, population_size*num_populations, fitness_function, fitness_function.n_param, jrandom.fold_in(key, 0))
    
    elif program == "Dynamic":
        fitness_function = dynamic_evaluate.Evaluator(env, state_size, dt0)

        # Define the expressions for the layers
        layer_expressions = [Expression(obs_size=env.n_obs, state_size=state_size, control_size=env.n_control, target_size=env.n_targets, unary_functions = unary_functions, operators_prob=operators_prob), 
                            Expression(obs_size=0, state_size=state_size, control_size=0, target_size=env.n_targets, unary_functions = unary_functions,
                                condition=lambda self, tree: sum([leaf in self.state_variables for leaf in jax.tree_util.tree_leaves(tree)])==0, operators_prob = operators_prob)]

        layer_sizes = jnp.array([state_size, env.n_control])
        assert len(layer_sizes) == len(layer_expressions), "There is not a set of expressions for every type of layer"
        
        strategy = GeneticProgramming(num_generations, population_size, fitness_function, layer_expressions, layer_sizes, 
                                num_populations = num_populations, state_size = state_size, pool_size = pool_size)
        
    elif program == "LQG":
        assert env_string == "HO", "LQG is only implemented for the Harmonic Oscillator"
        fitness_function = LQG_evaluate.Evaluator(env, dt0)
        _,_,_, fitness = fitness_function(data)

        return fitness

    # Initialize the population
    population = strategy.initialize_population(init_key)

    # Run the algorithm for a number of generations
    for g in range(num_generations):
        fitnesses, population = strategy.evaluate_population(population, data)
        
        best_fitness, best_solution = strategy.get_statistics(g)
        if program == "NDE":
            print(f"In generation {g+1}, best fitness = {best_fitness}")
        else:
            print(f"In generation {g+1}, best fitness = {best_fitness}, best solution = {best_solution}")

        key, sample_key = jrandom.split(key)
        population = strategy.sample_population(sample_key, population)

    return strategy.get_statistics()

if __name__ == '__main__':
    algorithms = ["Static", "Dynamic", "Random", "NDE", "LQG"]
    seed = 1

    exp1_args = ["HO", "Constant", 2, 30, 5, 0.05, 0.3]
    exp2_args = ["HO", "Constant", 1, 50, 5, 0.05, 0.3]
    exp3_args = ["HO", "Different", 2, 100, 10, 0.05, 0.3]
    exp4_args = ["ACR", "Constant", 4, 50, 5, 0.05, 0.3]
    exp5_args = ["ACR", "Constant", 2, 50, 5, 0.05, 0.3]
    exp6_args = ["ACR", "Constant", 4, 50, 5, 0.05, 0.3]
    exp7_args = ["CSTR", "Different", 2, 100, 10, 0.05, 0.5]

    # best_fitness, best_solutions = run(seed, "Static", exp1_args)
    # best_fitness, best_solutions = run(seed, "Dynamic", exp1_args)
    # best_fitness, best_solutions = run(seed, "Random", exp1_args)
    # best_fitness = run(seed, "NDE", exp1_args)
    fitness = run(seed, "LQG", exp1_args)