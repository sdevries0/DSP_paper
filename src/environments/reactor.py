from environments.environment_base import EnvironmentBase
import jax.numpy as jnp
import jax
import jax.random as jrandom
import diffrax
from jaxtyping import Array

class StirredTankReactor(EnvironmentBase):
    def __init__(self, process_noise: float = 0.0, obs_noise: float = 0.0, n_obs: int = 3, n_targets: int = 1, max_control: int = 300, external_f: callable = lambda t: 0.0, max_target = 500) -> None:
        self.n_var = 3
        self.n_control_inputs = 1
        self.n_dim = 1
        self.n_targets = n_targets
        self.init_lower_bounds = jnp.array([275, 350, 0.5])
        self.init_upper_bounds = jnp.array([300, 375, 1.0])
        self.max_control = max_control
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control_inputs, self.n_dim, n_obs)

        self.Q = jnp.array([[0, 0, 0], [0, 0.01, 0], [0, 0, 0]])
        self.r = jnp.array([[0.0001]])
        self.external_f = external_f
        self.max_target = max_target

    def initialize_parameters(self, params, ts):
        Vol, Cp, dHr, UA, q, Tf, Tcf, Volc = params
        self.Ea  = 72750     # activation energy J/gmol
        self.R   = 8.314     # gas constant J/gmol/K
        self.k0  = 7.2e10    # Arrhenius rate constant 1/min
        self.Vol = Vol       # Volume [L]
        self.Cp  = Cp        # Heat capacity [J/g/K]
        self.dHr = dHr       # Enthalpy of reaction [J/mol]
        self.UA  = UA        # Heat transfer [J/min/K]
        self.q = q           # Flowrate [L/min]
        self.Cf = 1.0        # Inlet feed concentration [mol/L]
        self.Tf = diffrax.LinearInterpolation(ts, Tf)  # Inlet feed temperature [K]
        self.Tcf = Tcf       # Coolant feed temperature [K]
        self.Volc = Volc       # Cooling jacket volume

        self.k = lambda T: self.k0*jnp.exp(-self.Ea/self.R/T)

        self.G = jnp.eye(self.n_var)*jnp.array([6, 6, 0.05])
        self.V = self.process_noise*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)*(jnp.array([15,15,0.1])[:self.n_obs])

        self.external_influence = diffrax.LinearInterpolation(ts, jax.vmap(self.external_f)(ts))

    def sample_param_change(self, key: jrandom.PRNGKey, batch_size: int, ts: Array, low: float, high: float) -> Array:
        """
        Samples parameter changes over time.

        Parameters
        ----------
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.
        batch_size : int
            Number of samples.
        ts : :class:`jax.Array`
            Time steps.
        low : float
            Lower bound for sampling.
        high : float
            Upper bound for sampling.

        Returns
        -------
        :class:`jax.Array`
            Sampled parameter values.
        """
        init_key, decay_key = jrandom.split(key)
        decay_factors = jrandom.uniform(decay_key, shape=(batch_size,), minval=1.01, maxval=1.02)
        init_values = jrandom.uniform(init_key, shape=(batch_size,), minval=low, maxval=high)
        values = jax.vmap(lambda v, d, t: v * (d ** t), in_axes=[0, 0, None])(init_values, decay_factors, ts)
        return values

    def sample_params(self, batch_size: int, mode: str, ts, key: jrandom.PRNGKey):
        """
        Samples parameters for the environment.

        Parameters
        ----------
        batch_size : int
            Number of samples.
        mode : str
            Sampling mode. Options are "Constant", "Different", "Changing".
        ts : :class:`jax.Array`
            Time steps.
        key : :class:`jax.random.PRNGKey`
            Random key for sampling.

        Returns
        -------
        tuple of :class:`jax.Array`
            Sampled parameters.
        """
        if mode == "Constant":
            Vol = 100 * jnp.ones(batch_size)
            Cp = 239 * jnp.ones(batch_size)
            dHr = -5.0e4 * jnp.ones(batch_size)
            UA = 5.0e4 * jnp.ones(batch_size)
            q = 100 * jnp.ones(batch_size)
            Tf = 300 * jnp.ones((batch_size, ts.shape[0]))
            Tcf = 300 * jnp.ones(batch_size)
            Volc = 20.0 * jnp.ones(batch_size)
        elif mode == "Different":
            keys = jrandom.split(key, 8)
            Vol = jrandom.uniform(keys[0], (batch_size,), minval=75, maxval=150)
            Cp = jrandom.uniform(keys[1], (batch_size,), minval=200, maxval=350)
            dHr = jrandom.uniform(keys[2], (batch_size,), minval=-55000, maxval=-45000)
            UA = jrandom.uniform(keys[3], (batch_size,), minval=25000, maxval=75000)
            q = jrandom.uniform(keys[4], (batch_size,), minval=75, maxval=125)
            Tf = jnp.repeat(jrandom.uniform(keys[5], (batch_size,), minval=300, maxval=350)[:, None], ts.shape[0], axis=1)
            Tcf = jrandom.uniform(keys[6], (batch_size,), minval=250, maxval=300)
            Volc = jrandom.uniform(keys[7], (batch_size,), minval=10, maxval=30)
        elif mode == "Changing":
            keys = jrandom.split(key, 8)
            Vol = jrandom.uniform(keys[0], (batch_size,), minval=75, maxval=150)
            Cp = jrandom.uniform(keys[1], (batch_size,), minval=200, maxval=350)
            dHr = jrandom.uniform(keys[2], (batch_size,), minval=-55000, maxval=-45000)
            UA = jrandom.uniform(keys[3], (batch_size,), minval=25000, maxval=75000)
            q = jrandom.uniform(keys[4], (batch_size,), minval=75, maxval=125)
            Tf = self.sample_param_change(keys[5], batch_size, ts, 300, 350)
            Tcf = jrandom.uniform(keys[6], (batch_size,), minval=250, maxval=300)
            Volc = jrandom.uniform(keys[7], (batch_size,), minval=10, maxval=30)
        return Vol, Cp, dHr, UA, q, Tf, Tcf, Volc

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = jrandom.uniform(init_key, shape=(batch_size, self.n_var), minval=self.init_lower_bounds, maxval=self.init_upper_bounds)
        targets = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=400, maxval=self.max_target)
        return x0, targets
    
    def f_obs(self, key, t_x):
        _, out = super().f_obs(key, t_x)
        # out = jnp.array([out[0], out[1], jnp.clip(out[2], 0, 1)])
        return key, out
    
    def drift(self, t, state, args):
        Tc, T, c = state
        control = jnp.squeeze(args)
        control = jnp.clip(control, 0, self.max_control)

        dc = (self.q / self.Vol) * (self.Cf - c) - self.k(T) * c
        dT = (self.q / self.Vol) * (self.Tf.evaluate(t) - T) + (-self.dHr / self.Cp) * self.k(T) * c + (self.UA / self.Vol / self.Cp) * (Tc - T) + self.external_influence.evaluate(t)
        dTc = (control / self.Volc) * (self.Tcf - Tc) + (self.UA / self.Volc / self.Cp) * (T - Tc)

        return jnp.array([dTc, dT, dc])

    def diffusion(self, t, state, args):
        return self.V

    def fitness_function(self, state, control, targets, ts):
        x_d = jax.vmap(lambda tar: jnp.array([0, tar, 0]))(targets)
        costs = jax.vmap(lambda _state, _u, _x_d: (_state - _x_d).T @ self.Q @ (_state - _x_d) + (_u) @ self.r @ (_u))(state, control, x_d)
        return jnp.sum(costs)

    def terminate_event(self, state, **kwargs):
        # return jnp.where(jnp.any(jnp.isinf(state.y) + jnp.isnan(state.y)), -1.0, 1.0)
        return False