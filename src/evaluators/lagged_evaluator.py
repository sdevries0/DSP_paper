import jax
import jax.numpy as jnp
import jax.random as jrandom
import time
from typing import Sequence, Tuple
import copy
import diffrax

class Evaluator:
    def __init__(self, env, state_size) -> None:
        self.env = env
        self.max_fitness = 1e6
        self.latent_size = env.n_var
        self.lag_number = 3

    def __call__(self, model, data) -> float:
        _, _, _, fitness = self.evaluate_model(model, data)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_model(self, model, data):
        "Evaluate a tree by simulating the environment and controller as a coupled system"
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0])(model, *data)
    
    def evaluate_control_loop(self, model: Tuple, x0: Sequence[float], ts: Sequence[float], target: float, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple):
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation 
        of the control is defined by the set of trees
        Inputs:
            model (NetworkTrees): Model with trees for the hidden state and readout
            x0 (float): Initial state of the system
            ts (Array[float]): time points on which the controller is evaluated
            target (float): Target position that the system should reach
            key (PRNGKey)
            params (Tuple[float]): Parameters that define the system

        Returns:
            xs (Array[float]): States of the system at every time point
            ys (Array[float]): Observations of the system at every time point
            us (Array[float]): Control of the model at every time point
            activities (Array[float]): Activities of the hidden state of the model at every time point
            fitness (float): Fitness of the model 
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        policy = model[0]
        dt = ts[1] - ts[0]
        
        def solve(carry, t):
            state, noise_key, buffer = carry
            noise_key, _key = jrandom.split(noise_key)
            _, y = env.f_obs(obs_noise_key, (t, state))
            buffer = jnp.concatenate([buffer[1:], jnp.expand_dims(y, axis=0)])
            u = policy({"y":jnp.ravel(buffer), "tar":target})
            new_state = state + dt * env.drift(t, state, u) + jnp.sqrt(dt) * jrandom.normal(_key, (env.n_var,)) @ env.diffusion(t, state, 0)
            
            new_carry = (new_state, noise_key, buffer)
            return new_carry, (new_state, y, u)
        
        buffer = jnp.zeros((self.lag_number, env.n_obs))
        
        _, (xs, ys, us) = jax.lax.scan(solve, (x0, process_noise_key, buffer), ts)
        xs = xs[::10]
        ys = ys[::10]
        us = us[::10]

        fitness = env.fitness_function(xs, us, target, ts)

        return xs, ys, us, fitness
    
    