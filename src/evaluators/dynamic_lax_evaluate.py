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
        self.state_size = state_size
        self.latent_size = env.n_var*env.n_dim

    def __call__(self, model, data) -> float:
        _, _, _, _, fitness = self.evaluate_model(model, data)

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

        state_equation, readout = model

        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        targets = diffrax.LinearInterpolation(ts, jnp.hstack([t*jnp.ones(int(ts.shape[0]//target.shape[0])) for t in target]))

        dt = ts[1] - ts[0]
        
        def solve(carry, t):
            x_a, noise_key = carry
            state = x_a[:self.latent_size]
            activities = x_a[self.latent_size:]
            noise_key, _key = jrandom.split(noise_key)

            tar = jnp.array([targets.evaluate(t)])

            _, y = env.f_obs(obs_noise_key, (t, state))
            u = readout({"y":y, "a":activities, "tar":tar})
            
            new_state = state + dt * env.drift(t, state, u) + jnp.sqrt(dt) * jrandom.normal(_key, (env.n_var,)) @ env.diffusion(t, state, 0)
            new_activities = activities + dt*state_equation({"y":y, "a":activities, "u":u, "tar":tar})
            
            new_carry = (jnp.concatenate([new_state, new_activities]), noise_key)
            return new_carry, (new_state, y, u, new_activities)
        
        _, (xs, ys, us, activities) = jax.lax.scan(solve, (_x0, process_noise_key), ts)
        xs = xs[::10]
        ys = ys[::10]
        us = us[::10]
        activities = activities[::10]
        targets_evaluated = targets.evaluate(ts)[::10]

        fitness = env.fitness_function(xs, us, targets_evaluated, ts)

        return xs, ys, us, activities, fitness