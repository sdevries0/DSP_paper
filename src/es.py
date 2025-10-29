import jax
import jax.numpy as jnp
from evosax.algorithms import SimpleES
import dataclasses

class MyES():
    def __init__(self, num_generations, population_size, fitness_function, sigma_init) -> None:
        self.num_generations = num_generations
        self.population_size = population_size
        self.fitness_function = fitness_function
        c_std = 0.1
        self.solution = jnp.zeros(self.fitness_function.n_param)

        self.cma_strategy = SimpleES(population_size = population_size, solution = self.solution)
        self.params = self.cma_strategy.default_params
        self.params = dataclasses.replace(self.params, std_init=sigma_init, c_std=c_std, weights = jnp.concatenate([jnp.ones(5)*0.1, jnp.zeros(self.fitness_function.n_param-5)]))

    def reset(self, key):
        self.current_generation = 0
        self.best_solutions = []
        self.best_fitnesses = jnp.zeros(self.num_generations)
        self.cma_state = self.cma_strategy.init(key, self.solution, self.params)

    def initialize_population(self, key):
        population, self.cma_state = self.cma_strategy.ask(key, self.cma_state, self.params)
        return population

    def evaluate_population(self, population, data, key):
        fitnesses = jax.vmap(self.fitness_function, in_axes=[0,None])(population, data)
        self.cma_state, _ = self.cma_strategy.tell(key, population, fitnesses, self.cma_state, self.params)

        best_idx = jnp.argmin(fitnesses)
        best_fitness_of_g = fitnesses[best_idx]
        best_solution_of_g = population[best_idx]

        if self.current_generation == 0:
            best_fitness = best_fitness_of_g
            best_solution = best_solution_of_g
        else:
            best_fitness = self.best_fitnesses[self.current_generation - 1]
            best_solution = self.best_solutions[self.current_generation - 1]

            if best_fitness_of_g < best_fitness:
                best_fitness = best_fitness_of_g
                best_solution = best_solution_of_g

        self.best_fitnesses = self.best_fitnesses.at[self.current_generation].set(best_fitness)
        self.best_solutions.append(best_solution)

        self.current_generation += 1
        return fitnesses, population

    def evolve_population(self, key, population):
        population, self.cma_state = self.cma_strategy.ask(key, self.cma_state, self.params)
        return population
    
    def get_statistics(self, generation = None):
        if generation is not None:
            return self.best_fitnesses[generation], self.best_solutions[generation]
        else:
            return self.best_fitnesses, self.best_solutions