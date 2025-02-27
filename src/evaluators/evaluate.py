import jax
import jax.numpy as jnp
import jax.random as jrandom
import time
from typing import Sequence, Tuple
import copy
import diffrax

class Evaluator:
    def __init__(self, env, state_size, dt0) -> None:
        self.env = env
        self.max_fitness = 1e6
        self.state_size = state_size
        self.latent_size = env.n_var*env.n_dim
        self.dt0 = dt0

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

        #Define state equation
        def _drift(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]

            _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
            u = readout({"y":y, "a":a, "tar":target}) #Readout control from hidden state

            dx = env.drift(t, x, u) #Apply control to system and get system change
            da = state_equation({"y":y, "a":a, "u":u, "tar":target}) #Compute hidden state updates
            return jnp.concatenate([dx, da])
        
        #Define diffusion
        def _diffusion(t, x_a, args):
            x = x_a[:self.latent_size]
            a = x_a[self.latent_size:]
            # _, y = env.f_obs(obs_noise_key, (t, x))
            # u = readout({"y":y, "a":a, "tar":target})
            # u = a
            return jnp.concatenate([env.diffusion(t, x, jnp.array([0])), jnp.zeros((self.state_size, self.latent_size))]) #Only the system is stochastic
        
        solver = diffrax.Euler()
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key)
        system = diffrax.MultiTerm(diffrax.ODETerm(_drift), diffrax.ControlTerm(_diffusion, brownian_motion))
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], self.dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=16**5, discrete_terminating_event=self.env.terminate_event
        )

        xs = sol.ys[:,:self.latent_size]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        activities = sol.ys[:,self.latent_size:]
        us = jax.vmap(lambda y, a, tar: readout({"y":y, "a":a, "tar":tar}), in_axes=[0,0,None])(ys, activities, target)

        fitness = env.fitness_function(xs, us, target, ts)

        return xs, ys, us, activities, fitness