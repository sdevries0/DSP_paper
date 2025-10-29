import jax
import jax.numpy as jnp
import jax.random as jrandom
import time
from typing import Sequence, Tuple
import copy
import diffrax

class LaggedEvaluator:
    def __init__(self, env, timesteps, dt0: float) -> None:
        self.env = env
        self.max_fitness = 1e6
        self.latent_size = env.n_var
        self.lag_number = timesteps
        self.dt0 = dt0

    def __call__(self, model, data, tree_evaluator) -> float:
        _, _, _, fitness = self.evaluate_model(model, data, tree_evaluator)

        return jnp.mean(fitness)
    
    def evaluate_model(self, model, data, tree_evaluator):
        "Evaluate a tree by simulating the environment and controller as a coupled system"
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0, None])(model, *data, tree_evaluator)
    
    def evaluate_control_loop(self, model: Tuple, x0: Sequence[float], ts: Sequence[float], target: float, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple, tree_evaluator):
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
        dt = ts[1] - ts[0]

        if target.shape[0] == 0:
            targets = diffrax.LinearInterpolation(ts, jnp.zeros_like(ts))
        else:
            targets = diffrax.LinearInterpolation(ts, jnp.hstack([t*jnp.ones(int(ts.shape[0]//target.shape[0])) for t in target]))
        
        def solve(carry, t):
            state, noise_key, buffer = carry
            noise_key, _key = jrandom.split(noise_key)
            _, y = env.f_obs(obs_noise_key, (t, state))
            buffer = jnp.concatenate([buffer[1:], jnp.expand_dims(y, axis=0)])

            u = tree_evaluator(model, jnp.concatenate([jnp.ravel(buffer), jnp.atleast_1d(targets.evaluate(t))]))
            new_state = state + dt * env.drift(t, state, u)[0] + jnp.sqrt(dt) * jrandom.normal(_key, (env.n_var,)) @ env.diffusion(t, state, 0)
            
            new_carry = (new_state, noise_key, buffer)
            return new_carry, (new_state, y, u)
        
        buffer = jnp.zeros((self.lag_number, env.n_obs))
        
        _, (xs, ys, us) = jax.lax.scan(solve, (x0, process_noise_key, buffer), ts)

        ratio = 10
        xs = xs[::ratio]
        ys = ys[::ratio]
        us = us[::ratio]
        _targets = targets.evaluate(ts)[::ratio]

        fitness = env.fitness_function(xs, us, _targets, ts[::ratio])

        return xs, ys, us, fitness
    
    