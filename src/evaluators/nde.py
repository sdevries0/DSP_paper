import jax
import jax.numpy as jnp
import jax.random as jr
import time
from jax.random import PRNGKey
from typing import Sequence, Tuple
import copy
import diffrax

class NDE_Evaluator:
    def __init__(self, env, state_size, dt0, solver=diffrax.Euler(), max_steps: int = 16**4, stepsize_controller: diffrax.AbstractStepSizeController = diffrax.ConstantStepSize()) -> None:
        self.env = env
        self.state_size = state_size
        self.max_fitness = 1e6
        self.obs_size = env.n_obs
        self.control_size = env.n_control_inputs
        self.latent_size = env.n_var*env.n_dim
        self.target_dim = env.n_targets
        self.solver = solver
        self.stepsize_controller = stepsize_controller

        self.parameter_reshaper = ParameterReshaper(obs_space = self.obs_size, latent_size = self.state_size, action_space = self.control_size, n_targets = self.env.n_targets)
        self.n_param = self.parameter_reshaper.total_parameters
        
        self.dt0 = dt0
        self.max_steps = max_steps

    def __call__(self, weights, data):
        _, _, _, _, fitness = self.evaluate_model(weights, data)

        nan_or_inf =  jax.vmap(lambda f: jnp.isinf(f) + jnp.isnan(f))(fitness)
        fitness = jnp.where(nan_or_inf, jnp.ones(fitness.shape)*self.max_fitness, fitness)
        fitness = jnp.mean(fitness)
        return jnp.clip(fitness,0,self.max_fitness)
    
    def evaluate_model(self, weights, data):
        "Evaluate a tree by simulating the environment and controller as a coupled system"
        
        model = RNN(*self.parameter_reshaper(weights))

        #Run coupled differential equations of state and control and get fitness of the model
        return jax.vmap(self.evaluate_control_loop, in_axes=[None, 0, None, 0, 0, 0, 0])(model, *data)

    def evaluate_control_loop(self, model, x0: Sequence[float], ts: Sequence[float], target: float, process_noise_key: PRNGKey, obs_noise_key: PRNGKey, params: Tuple):
        """Solves the coupled differential equation of the system and controller. The differential equation of the system is defined in the environment and the differential equation of the control is defined by the set of trees
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
        
        solver = self.solver
        saveat = diffrax.SaveAt(ts=ts)
        _x0 = jnp.concatenate([x0, jnp.zeros(self.state_size)])

        if target.shape[0] == 0:
            targets = diffrax.LinearInterpolation(ts, jnp.zeros_like(ts))
        else:
            targets = diffrax.LinearInterpolation(ts, jnp.hstack([t*jnp.ones(int(ts.shape[0]//target.shape[0])) for t in target]))

        brownian_motion = diffrax.UnsafeBrownianPath(shape=(self.latent_size,), key=process_noise_key, levy_area=diffrax.SpaceTimeLevyArea) #define process noise
        system = diffrax.MultiTerm(diffrax.ODETerm(self._drift), diffrax.ControlTerm(self._diffusion, brownian_motion))

        sol = diffrax.diffeqsolve(
            system, solver, ts[0], ts[-1], self.dt0, _x0, saveat=saveat, adjoint=diffrax.DirectAdjoint(), max_steps=self.max_steps, event=diffrax.Event(self.env.cond_fn_nan),
            args=(env, model, obs_noise_key, targets), stepsize_controller=self.stepsize_controller, throw=False
        )

        xs = sol.ys[:,:self.latent_size]
        _, ys = jax.lax.scan(env.f_obs, obs_noise_key, (ts, xs))
        activities = sol.ys[:,self.latent_size:]
        if self.target_dim==0:
            us = jax.vmap(lambda a: model.act(a))(activities)
        else:
            us = jax.vmap(lambda a, tar: model.act(a, tar))(activities, targets.evaluate(ts)[:,None])

        fitness = env.fitness_function(xs, us, targets.evaluate(ts), ts)

        return xs, ys, us, activities, fitness
    
    #Define state equation
    def _drift(self, t, x_a, args):
        env, model, obs_noise_key, target = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]

        _, y = env.f_obs(obs_noise_key, (t, x)) #Get observations from system
        if self.target_dim == 0:
            u = model.act(a)
            dx, _u = env.drift(t, x, u) #Apply control to system and get system change
            da = model.update(y, a, _u)
        else:
            u = model.act(a,  jnp.atleast_1d(target.evaluate(t)))
            dx, _u = env.drift(t, x, u) #Apply control to system and get system change
            da = model.update(y, a, _u, jnp.atleast_1d(target.evaluate(t)))
        
         #Compute hidden state updates

        return jnp.concatenate([dx, da])
    
    #Define diffusion
    def _diffusion(self, t, x_a, args):
        env, model, obs_noise_key, target = args
        x = x_a[:self.latent_size]
        a = x_a[self.latent_size:]

        return jnp.concatenate([env.diffusion(t, x, jnp.zeros(self.control_size)), jnp.zeros((self.state_size, self.latent_size))]) #Only the system is stochastic
    
class RNN:
    def __init__(self, input_layer, action_layer) -> None:
        self.input_layer = input_layer
        self.action_layer = action_layer
        self.bias = jnp.ones(1)

    def update(self, y, a, u, target=None):
        if target == None:
            x = jnp.concatenate([y, u, a, self.bias])
        else:
            x = jnp.concatenate([y, u, a, target, self.bias])
        x = jnp.tanh(self.input_layer@x)
        return x
    
    def act(self, a, target=None):
        if target == None:
            x = jnp.concatenate([a, self.bias])
        else:
            x = jnp.concatenate([a, target, self.bias])

        return self.action_layer@x
    
class ParameterReshaper:
    def __init__(self, obs_space, latent_size, action_space, n_targets):
        self.input_layer_shape = (obs_space + action_space + latent_size + n_targets + 1, latent_size)

        self.action_layer_shape = (latent_size + n_targets + 1, action_space)

        self.nr_params_input = self.input_layer_shape[0] * self.input_layer_shape[1]
        self.nr_params_action = self.action_layer_shape[0] * self.action_layer_shape[1]

        self.total_parameters = self.nr_params_input + self.nr_params_action
                            
    def __call__(self, params):
        assert params.shape[0] == self.total_parameters

        layers = []

        w = params[:self.nr_params_input].reshape(self.input_layer_shape[1], self.input_layer_shape[0])
        layers.append(w)

        w = params[self.nr_params_input:].reshape(self.action_layer_shape[1], self.action_layer_shape[0])
        layers.append(w)

        return layers