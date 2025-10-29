import os
import sys
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import time
from typing import Tuple, Callable
import diffrax

# Import modern Kozax components
from kozax.genetic_programming import GeneticProgramming
from static_evaluate import StaticEvaluator
from dynamic_evaluate import DynamicEvaluator
from lagged_evaluator import LaggedEvaluator

# Import control environments from Kozax (with fallbacks if not available)
from kozax.environments.control_environments.harmonic_oscillator import HarmonicOscillator

def get_data(key, env, batch_size, dt, T, param_setting):
    """Generate training data for the control environment"""
    init_key, noise_key1, noise_key2, param_key = jr.split(key, 4)
    
    # Generate initial states and targets
    x0, targets = env.sample_init_states(batch_size, init_key)
    
    process_noise_keys = jr.split(noise_key1, batch_size)
    obs_noise_keys = jr.split(noise_key2, batch_size)
    ts = jnp.arange(0, T, dt)
    
    # Generate parameters
    params = env.sample_params(batch_size, param_setting, ts, param_key)
    
    return x0, ts, targets, process_noise_keys, obs_noise_keys, params

def run(program, obs_noise):

    # Define the parameters of the algorithm
    population_size = 100
    num_populations = 5
    num_generations = 50
    state_size = 2
    dt = 0.2
    batch_size = 32

    # Define the environment using modern Kozax
    env = HarmonicOscillator(0.05, obs_noise, 1)  # Use default parameters for now

    operator_list = [
        ("+", lambda x, y: jnp.add(x, y), 2, 0.5),
        ("*", lambda x, y: jnp.multiply(x, y), 2, 0.3),
        ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
        ("/", lambda x, y: jnp.divide(x, y), 2, 0.1),
    ]

    dt0 = 0.02
    max_steps = 2000
    T = 30

    if program == "Static":
        variable_list = [["y" + str(i) for i in range(env.n_obs)]]
        if env.n_targets > 0:
            variable_list[0].append("tar")
        
        # Define layer sizes (output size = control dimension)
        layer_sizes = jnp.array([env.n_control_inputs])
        
        fitness_function = StaticEvaluator(env, dt0, solver=diffrax.GeneralShARK(), max_steps = max_steps)

        strategy = GeneticProgramming(
            fitness_function=fitness_function,
            num_generations=num_generations,
            population_size=population_size,
            operator_list=operator_list,
            variable_list=variable_list,
            num_populations=num_populations,
            layer_sizes=layer_sizes,
            complexity_objective=True,
            max_init_depth=4,
        )

        name = "GP-S"

    elif program == "Dynamic":
        variable_list = [["y" + str(i) for i in range(env.n_obs)] + ["a" + str(i) for i in range(state_size)] + ["u" + str(i) for i in range(env.n_control_inputs)], ["a" + str(i) for i in range(state_size)]]
        if env.n_targets > 0:
            for var_list in variable_list:
                var_list.append("tar")

        # Define layer sizes (output size = control dimension)
        layer_sizes = jnp.array([state_size, env.n_control_inputs])
        
        fitness_function = DynamicEvaluator(env, state_size, dt0, solver=diffrax.GeneralShARK(), max_steps = max_steps)

        strategy = GeneticProgramming(
            fitness_function=fitness_function,
            num_generations=num_generations,
            population_size=population_size,
            operator_list=operator_list,
            variable_list=variable_list,
            num_populations=num_populations,
            layer_sizes=layer_sizes,
            complexity_objective=True,
            max_init_depth=4,
        )

        name = "GP-D"

    elif program == "Lag":
        lag_steps = 5
        variable_list = [["y" + str(i) for i in range(lag_steps)]]
        if env.n_targets > 0:
            for var_list in variable_list:
                var_list.append("tar")

        # Define layer sizes (output size = control dimension)
        layer_sizes = jnp.array([env.n_control_inputs])

        dt = dt/10
        
        fitness_function = LaggedEvaluator(env, lag_steps, dt0)

        strategy = GeneticProgramming(
            fitness_function=fitness_function,
            num_generations=num_generations,
            population_size=population_size,
            operator_list=operator_list,
            variable_list=variable_list,
            num_populations=num_populations,
            layer_sizes=layer_sizes,
            complexity_objective=True,
            max_init_depth=5,
            max_nodes=25,
        )

        name = "GP-L"

    for seed in range(20):
        print("seed", seed)
        key = jr.PRNGKey(seed)
        key, init_key, data_key = jr.split(key, 3)

        best_fitnesses = []

        # Get the data
        data = get_data(data_key, env, batch_size, dt, T, "Constant")        
    
        # Use the fit method for training
        strategy.reset()
        
        # Warm up JIT functions with actual data shapes
        population = strategy.initialize_population(init_key)
        strategy._warm_up_jit_functions(population, data)

        start = time.time()

        for g in range(num_generations):
            key, eval_key, sample_key = jr.split(key, 3)
            # Evaluate the population on the data, and return the fitness
            fitness, population = strategy.evaluate_population(population, data, eval_key)

            best_fitnesses.append(jnp.min(fitness))

            if (g%5)==0:
                print("Generation:", g)
                strategy.print_pareto_front()

            # Evolve the population until the last generation. The fitness should be given to the evolve function.
            if g < (num_generations-1):
                population = strategy.evolve_population(population, fitness, sample_key)

        end = time.time()
        # Save timing results
        os.makedirs(f'../results/DSP_paper/LagExp/{obs_noise}/{name}/', exist_ok=True)
        np.save(f'../results/DSP_paper/LagExp/{obs_noise}/{name}/time_{seed}.npy', end - start)

        # Get the best solution from the pareto front
        best_idx = jnp.argmin(strategy.pareto_front[0])
        best_fitness = strategy.pareto_front[0][best_idx]  # Best fitness
        best_solution = strategy.pareto_front[1][best_idx]  # Best solution
        best_solution_str = strategy.expression_to_string(best_solution)
        
        print(f"Final best fitness: {best_fitness}")
        print(f"Final best solution: {best_solution_str}")

        np.save(f'../results/DSP_paper/LagExp/{obs_noise}/{name}/best_fitness_{seed}.npy', best_fitnesses)
        np.save(f'../results/DSP_paper/LagExp/{obs_noise}/{name}/best_solutions_{seed}.npy', strategy.pareto_front[1])

if __name__ == '__main__':
    algorithms = ["Static", "Lag", "Dynamic"]

    method = int(sys.argv[1])
    obs_noise = float(sys.argv[2])
    
    # Run the selected experiment
    run(algorithms[method], obs_noise)