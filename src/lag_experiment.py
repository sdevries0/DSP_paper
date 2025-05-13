import os
import sys

print("nr of cpus", os.cpu_count())
import jax
print(jax.devices())

import jax.numpy as jnp
import jax.random as jrandom

from miscellaneous.expression import Expression

from algorithms.genetic_programming import GeneticProgramming

from environments.harmonic_oscillator import HarmonicOscillator

import evaluators.dynamic_lax_evaluate as dynamic_lax_evaluate
import evaluators.lagged_evaluator as lagged_evaluate

def get_data(key, env, batch_size, dt, T, param_setting):
    """
    Get data for the environment"""
    init_key, noise_key1, noise_key2, param_key = jrandom.split(key, 4)
    x0, targets = env.sample_init_states(batch_size, init_key)
    process_noise_keys = jrandom.split(noise_key1, batch_size)
    obs_noise_keys = jrandom.split(noise_key2, batch_size)
    ts = jnp.arange(0,T,dt)

    params = env.sample_params(batch_size, param_setting, ts, param_key)
    return x0, ts, targets, process_noise_keys, obs_noise_keys, params

def run(seed, program, obs_noise):
    key = jrandom.PRNGKey(seed)
    init_key, data_key = jrandom.split(key)

    # Parameters
    population_size = 100
    num_generations = 5
    num_populations = 2
    state_size = 2
    T = 30
    dt = 0.02
    pool_size = os.cpu_count()
    batch_size = 4
    sigma = 0.05

    param_setting = "Constant"
    operators_prob = jnp.array([0.5, 0.3, 0.5, 0.1, 0.1])

    # Environment
    env = HarmonicOscillator(sigma, obs_noise, n_obs=1)

    # Program
    if program == "Lagged":
        fitness_function = lagged_evaluate.Evaluator(env, state_size)

        # Define the expressions for the layers
        layer_expressions = [Expression(obs_size=env.n_obs*3, target_size=env.n_targets, unary_functions = [], operators_prob=operators_prob)]

        layer_sizes = jnp.array([env.n_control])
        assert len(layer_sizes) == len(layer_expressions), "There is not a set of expressions for every type of layer"

        strategy = GeneticProgramming(num_generations, population_size, fitness_function, layer_expressions, layer_sizes, 
                                num_populations = num_populations, state_size = state_size, pool_size = pool_size)
    
    elif program == "Dynamic":
        fitness_function = dynamic_lax_evaluate.Evaluator(env, state_size)

        # Define the expressions for the layers
        layer_expressions = [Expression(obs_size=env.n_obs, state_size=state_size, control_size=env.n_control, target_size=env.n_targets, unary_functions = [], operators_prob=operators_prob), 
                            Expression(obs_size=0, state_size=state_size, control_size=0, target_size=env.n_targets, unary_functions = [],
                                condition=lambda self, tree: sum([leaf in self.state_variables for leaf in jax.tree_util.tree_leaves(tree)])==0, operators_prob = operators_prob)]

        layer_sizes = jnp.array([state_size, env.n_control])
        assert len(layer_sizes) == len(layer_expressions), "There is not a set of expressions for every type of layer"
        
        strategy = GeneticProgramming(num_generations, population_size, fitness_function, layer_expressions, layer_sizes, 
                                num_populations = num_populations, state_size = state_size, pool_size = pool_size)

    # Initialize the population
    population = strategy.initialize_population(init_key)

    # Get the data
    data = get_data(data_key, env, batch_size, dt, T, param_setting)

    # Run the algorithm for a number of generations
    for g in range(num_generations):
        fitnesses, population = strategy.evaluate_population(population, data)
        
        best_fitness, best_solution = strategy.get_statistics(g)
        print(f"In generation {g+1}, best fitness = {best_fitness}, best solution = {best_solution}")

        key, sample_key = jrandom.split(key)
        population = strategy.sample_population(sample_key, population)

    best_fitnesses, best_solutions = strategy.get_statistics()

    return best_fitnesses, best_solutions

if __name__ == '__main__':
    seed = 1
    best_fitness, best_solutions = run(seed, "Dynamic", obs_noise=0.5)
    # best_fitness, best_solutions = run(seed, "Lagged", obs_noise=0.5)