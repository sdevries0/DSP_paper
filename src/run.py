import os
import sys
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import time
import diffrax

# Import modern Kozax components
from kozax.genetic_programming import GeneticProgramming
from evaluators.static_evaluate import StaticEvaluator
from evaluators.dynamic_evaluate import DynamicEvaluator
from evaluators.nde import NDE_Evaluator
from es import MyES
from evaluators.lqg import LQG

# Import control environments from Kozax (with fallbacks if not available)
from environments.harmonic_oscillator import HarmonicOscillator
from environments.reactor import StirredTankReactor
from environments.acrobot import Acrobot, Acrobot2

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

def run(program, args, exp_id):
    env_string, param_setting, n_obs, num_generations, num_populations, process_noise, obs_noise, n_control = args

    # Define the parameters of the algorithm
    population_size = 100
    state_size = 2
    dt = 0.2
    batch_size = 32

    # Define the environment using modern Kozax
    if env_string == "HO":
        env = HarmonicOscillator(process_noise, obs_noise, n_obs)  # Use default parameters for now

        operator_list = [
            ("+", lambda x, y: jnp.add(x, y), 2, 0.5),
            ("*", lambda x, y: jnp.multiply(x, y), 2, 0.3),
            ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
            ("/", lambda x, y: jnp.divide(x, y), 2, 0.1),
        ]
        dt0 = 0.02
        max_steps = 2000
        T = 30
        nde_size = 5
        nde_sigma_init = 0.5

    elif env_string == "ACR":
        if n_control == 1:
            env = Acrobot(process_noise, obs_noise, n_obs)  # Use default parameters
        else:
            env = Acrobot2(process_noise, obs_noise, n_obs)

        operator_list = [
            ("+", lambda x, y: jnp.add(x, y), 2, 0.5),
            ("*", lambda x, y: jnp.multiply(x, y), 2, 0.3),
            ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
            ("/", lambda x, y: jnp.divide(x, y), 2, 0.1),
            ("sin", lambda x: jnp.sin(x), 1, 0.1),
            ("cos", lambda x: jnp.cos(x), 1, 0.1),
        ]
        dt0 = 0.02
        max_steps = 3000
        T = 50
        nde_size = 5
        nde_sigma_init = 5.0

    elif env_string == "CSTR":
        env = StirredTankReactor(process_noise, obs_noise, n_obs)  # Use default parameters

        operator_list = [
            ("+", lambda x, y: jnp.add(x, y), 2, 0.5),
            ("*", lambda x, y: jnp.multiply(x, y), 2, 0.3),
            ("-", lambda x, y: jnp.subtract(x, y), 2, 0.5),
            ("/", lambda x, y: jnp.divide(x, y), 2, 0.1),
            ("exp", lambda x: jnp.exp(jnp.clip(x, -10, 10)), 1, 0.1),
            ("log", lambda x: jnp.log(jnp.abs(x) + 1e-8), 1, 0.1),
        ]
        dt0 = 0.001
        max_steps = 40000
        T = 30
        nde_size = 10
        nde_sigma_init = 5.0

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
            device_type="cpu"  # Use CPU for compatibility
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
            device_type="cpu"  # Use CPU for compatibility
        )

        name = "GP-D"

    elif program == "NDE":
        name = "NDE"

        state_size = 5
        fitness_function = NDE_Evaluator(env, nde_size, dt0, solver=diffrax.GeneralShARK(), max_steps = max_steps)

        strategy = MyES(num_generations, population_size * num_populations, fitness_function, sigma_init=nde_sigma_init)

    elif program == "Random":
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
            max_init_depth=7,
            crossover_probability_factors = (0.0, 0.0),
            mutation_probability_factors = (0.0, 0.0),
            sample_probability_factors = (1.0, 1.0)
        )

        name = "RS"

    elif program == "LQG":   
        assert env_string == "HO", "LQG is only implemented for the Harmonic Oscillator"
        # Calculate LQG baseline cost
        _lqg = LQG(env, dt0, solver=diffrax.GeneralShARK(), max_steps = max_steps)
        fitnesses = np.zeros(20)

        for seed in range(20):
            print("seed", seed)
            key = jr.PRNGKey(seed)
            key, init_key, data_key = jr.split(key, 3)

            # Get the data
            data = get_data(data_key, env, batch_size, dt, T, param_setting)
            _, _, _, fitness = _lqg(data)
            fitnesses[seed] = fitness
            print(fitness)

            # Save timing results
            os.makedirs(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/LQG/', exist_ok=True)

        np.save(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/LQG.npy', fitnesses)
    
    else:
        raise ValueError(f"Unknown program type: {program}")

    if program in ["Static", "Dynamic", "Random"]:
        for seed in range(20):
            print("seed", seed)
            key = jr.PRNGKey(seed)
            key, init_key, data_key = jr.split(key, 3)

            best_fitnesses = []

            # Get the data
            data = get_data(data_key, env, batch_size, dt, T, param_setting)        
        
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
            os.makedirs(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/{name}/', exist_ok=True)
            np.save(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/{name}/time_{seed}.npy', end - start)

            # Get the best solution from the pareto front
            best_idx = jnp.argmin(strategy.pareto_front[0])
            best_fitness = strategy.pareto_front[0][best_idx]  # Best fitness
            best_solution = strategy.pareto_front[1][best_idx]  # Best solution
            best_solution_str = strategy.expression_to_string(best_solution)
            
            print(f"Final best fitness: {best_fitness}")
            print(f"Final best solution: {best_solution_str}")

            np.save(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/{name}/best_fitness_{seed}.npy', best_fitnesses)
            np.save(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/{name}/best_solutions_{seed}.npy', strategy.pareto_front[1])
            
    elif program == "NDE":
        for seed in range(20):
            print("seed", seed)
            key = jr.PRNGKey(seed)
            key, init_key, data_key = jr.split(key, 3)

            # Get the data
            data = get_data(data_key, env, batch_size, dt, T, param_setting)
        
            # Use the fit method for training
            key1, key2 = jr.split(init_key)
            strategy.reset(key1)
            population = strategy.initialize_population(key2)
            
            start = time.time()

            for g in range(num_generations):
                key, eval_key, sample_key = jr.split(key, 3)
                # Evaluate the population on the data, and return the fitness
                fitness, population = strategy.evaluate_population(population, data, eval_key)

                if (g%5)==0:
                    print("Generation:", g)
                    best_f, _ = strategy.get_statistics(g)
                    print(best_f)

                # Evolve the population until the last generation. The fitness should be given to the evolve function.
                if g < (num_generations-1):
                    population = strategy.evolve_population(sample_key, population)

            end = time.time()
            # Save timing results
            os.makedirs(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/{name}/', exist_ok=True)
            np.save(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/{name}/time_{seed}.npy', end - start)

            best_fitnesses, best_solutions = strategy.get_statistics()

            np.save(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/{name}/best_fitness_{seed}.npy', best_fitnesses)
            np.save(f'/home/sdevries/results/DSP_paper/Exp{exp_id}/{name}/best_solutions_{seed}.npy', jnp.array(best_solutions))

if __name__ == '__main__':
    algorithms = ["Static", "Dynamic", "NDE", "Random", "LQG"]

    method = int(sys.argv[1])
    exp_id = int(sys.argv[2]) - 1

    # Experiment configurations: [env, param_setting, n_obs, generations, populations, process_noise, obs_noise]
    exp1_args = ["HO", "Constant", 2, 50, 5, 0.05, 0.3, 1]
    exp2_args = ["HO", "Constant", 1, 50, 5, 0.05, 0.3, 1]
    exp3_args = ["HO", "Different", 2, 150, 10, 0.05, 0.3, 1]
    exp4_args = ["ACR", "Constant", 4, 50, 5, 0.05, 0.3, 1]
    exp5_args = ["ACR", "Constant", 2, 50, 5, 0.05, 0.3, 1]
    exp6_args = ["ACR", "Constant", 4, 50, 5, 0.05, 0.3, 2]
    exp7_args = ["CSTR", "Different", 2, 100, 10, 0.5, 0.5, 1]

    args = [exp1_args, exp2_args, exp3_args, exp4_args, exp5_args, exp6_args, exp7_args]
    
    # Run the selected experiment
    run(algorithms[method], args[exp_id], exp_id + 1)