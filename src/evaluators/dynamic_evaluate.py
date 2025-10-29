"""
kozax: Genetic programming framework in JAX

Copyright (c) 2024 

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import jax
from jax import Array
import jax.numpy as jnp
import jax.random as jrandom
import diffrax

from typing import Sequence, Tuple, Callable
import copy
    
class DynamicEvaluator:
    def __init__(self, env, state_size: int, dt0: float, solver=diffrax.Euler(), max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        """Evaluator for dynamic symbolic policies in control tasks

        Attributes:
            env: Environment on which the candidate is evaluated
            max_fitness: Max fitness which is assigned when a trajectory returns an invalid value
            state_size: Dimensionality of the hidden state
            obs_size: Dimensionality of the observations
            control_size: Dimensionality of the control
            latent_size: Dimensionality of the state of the environment
            dt0: Initial step size for integration
            solver: Solver used for integration
            max_steps: The maximum number of steps that can be used in integration
            stepsize_controller: Controller for the stepsize during integration
        """
        self.env = env
        self.state_size = state_size
        self.obs_size = env.n_obs
        self.control_size = env.n_control_inputs
        self.latent_size = env.n_var*env.n_dim
        self.dt0 = dt0
        self.solver = solver
        self.max_steps = max_steps
        self.stepsize_controller = stepsize_controller

    def __call__(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> float:
        """Evaluates the candidate on a task

        :param coefficients: The coefficients of the candidate
        :param nodes: The nodes and index references of the candidate
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees

        Returns: Fitness of the candidate
        """
        _, _, _, _, fitness = self.evaluate_candidate(candidate, data, tree_evaluator)

        return jnp.mean(fitness)
    
    def evaluate_candidate(self, candidate: Array, data: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, Array, float]:
        """Evaluates a candidate given a task and data

        :param candidate: Candidate that is evaluated
        :param data: The data required to evaluate the candidate
        :param tree_evaluator: Function for evaluating trees
        
        Returns: Predictions and fitness of the candidate
        """
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0, None])(candidate, *data, tree_evaluator)
    
    def evaluate_control_loop(self, candidate: Array, x0: Array, ts: Array, target: float, process_noise_key: jrandom.PRNGKey, obs_noise_key: jrandom.PRNGKey, params: Tuple, tree_evaluator: Callable) -> Tuple[Array, Array, Array, Array, float]:
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation 
        of the control is defined by the set of trees

        :param candidate: Candidate with trees for the hidden state and readout
        :param x0: Initial state of the system
        :param ts: time points on which the controller is evaluated
        :param target: Target position that the system should reach
        :param process_noise_key: Key to generate process noise
        :param obs_noise_key: Key to generate noisy observations
        :param params: Parameters that define the environment
        :param tree_evaluator: Function for evaluating trees

        Returns: States, observations, control, activities of the hidden state of the candidate and the fitness of the candidate.
        """
        env = copy.copy(self.env)
        env.initialize_parameters(params, ts)

        state_equation = candidate[:self.state_size]
        readout = candidate[self.state_size:]

        if target.shape[0] == 0:
            targets = diffrax.LinearInterpolation(ts, jnp.zeros_like(ts))
        else:
            targets = diffrax.LinearInterpolation(ts, jnp.hstack([t*jnp.ones(int(ts.shape[0]//target.shape[0])) for t in target]))
        
        solver = self.solver
        dt0 = self.dt0
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key, levy_area=diffrax.SpaceTimeLevyArea) #define process noise
        system = diffrax.MultiTerm(diffrax.ODETerm(self._drift), diffrax.ControlTerm(self._diffusion, brownian_motion))
        
        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps, 
            args=(env, state_equation, readout, obs_noise_key, targets, tree_evaluator), stepsize_controller=self.stepsize_controller, throw=False
        )

        xs = sol.ys[:,:self.latent_size]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        activities = sol.ys[:,self.latent_size:]
        us = jax.vmap(lambda y, a, tar: tree_evaluator(readout, jnp.concatenate([y, a, jnp.zeros(self.control_size), tar])), in_axes=[0,0,0])(ys, activities, targets.evaluate(ts)[:,None])

        fitness = env.fitness_function(xs, us, targets.evaluate(ts), ts)

        return xs, ys, us, activities, fitness
    
    def _drift(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, target, tree_evaluator = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]

        _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
        u = tree_evaluator(readout, jnp.concatenate([jnp.zeros(self.obs_size), a, jnp.zeros(self.control_size), jnp.atleast_1d(target.evaluate(t))]))

        dx, _u = env.drift(t, x, u) #Apply control to system and get system change
        da = tree_evaluator(state_equation, jnp.concatenate([y, a, _u, jnp.atleast_1d(target.evaluate(t))]))

        return jnp.concatenate([dx, da])
    
    def _diffusion(self, t, x_a, args):
        env, state_equation, readout, obs_noise_key, target, tree_evaluator = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]

        return jnp.concatenate([env.diffusion(t, x, jnp.array([0])), jnp.zeros((self.state_size, self.latent_size))]) #Only the system is stochastic