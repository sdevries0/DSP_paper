import jax
import jax.numpy as jnp
import jax.random as jrandom
import diffrax
from environments.environment_base import EnvironmentBase

class HarmonicOscillator(EnvironmentBase):
    def __init__(self, process_noise, obs_noise, n_obs = None):
        self.n_dim = 1
        self.n_var = 2
        self.n_control = 1
        self.n_targets = 1
        self.mu0 = jnp.zeros(self.n_var)
        self.P0 = jnp.eye(self.n_var)*jnp.array([3.0, 1.0])
        self.default_obs = 2
        super().__init__(process_noise, obs_noise, self.n_var, self.n_control, self.n_dim, n_obs if n_obs else self.default_obs)

        self.q = self.r = 0.5
        self.Q = jnp.array([[self.q,0],[0,0]])
        self.R = jnp.array([[self.r]])

    def sample_init_states(self, batch_size, key):
        init_key, target_key = jrandom.split(key)
        x0 = self.mu0 + jrandom.normal(init_key, shape=(batch_size, self.n_var))@self.P0
        targets = jrandom.uniform(target_key, shape=(batch_size, self.n_targets), minval=-3, maxval=3)
        return  x0, targets
    
    def sample_params(self, batch_size, mode, ts, key):
        omega_key, zeta_key, args_key = jrandom.split(key, 3)
        if mode == "Constant":
            # omegas = jnp.ones((batch_size, ts.shape[0]))
            # zetas = jnp.zeros((batch_size, ts.shape[0]))
            omegas = jnp.ones((batch_size))
            zetas = jnp.zeros((batch_size))
        elif mode == "Different":
            # omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.5, maxval=1.5)[:,None] * jnp.ones((batch_size,ts.shape[0]))
            # zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.0)[:,None] * jnp.ones((batch_size,ts.shape[0]))
            omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.0, maxval=2.0)
            zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.5)
        elif mode == "Switch":
            switch_times = jrandom.randint(args_key, shape=(batch_size,), minval=int(ts.shape[0]/4), maxval=int(3*ts.shape[0]/4))
            omegas = jnp.zeros((batch_size, ts.shape[0]))
            zetas = jnp.zeros((batch_size, ts.shape[0]))
            for i in range(batch_size):
                _key11, _key12 = jrandom.split(jrandom.fold_in(omega_key, i))
                omegas = omegas.at[i,:switch_times[i]].set(jrandom.uniform(_key11, shape=(), minval=0.5, maxval=1.5))
                omegas = omegas.at[i,switch_times[i]:].set(jrandom.uniform(_key12, shape=(), minval=0.5, maxval=1.5))

                _key21, _key22 = jrandom.split(jrandom.fold_in(zeta_key, i))
                zetas = zetas.at[i,:switch_times[i]].set(jrandom.uniform(_key21, shape=(), minval=0., maxval=1.))
                zetas = zetas.at[i,switch_times[i]:].set(jrandom.uniform(_key22, shape=(), minval=0., maxval=1.))

        elif mode == "Decay":
            decay_factors = jrandom.uniform(args_key, shape=(batch_size,2), minval=0.98, maxval=1.02)
            init_omegas = jrandom.uniform(omega_key, shape=(batch_size,), minval=0.5, maxval=1.5)
            init_zetas = jrandom.uniform(zeta_key, shape=(batch_size,), minval=0.0, maxval=1.0)
            omegas = jax.vmap(lambda o, d, t: o*(d**t), in_axes=[0, 0, None])(init_omegas, decay_factors[:,0], ts)
            zetas = jax.vmap(lambda z, d, t: z*(d**t), in_axes=[0, 0, None])(init_zetas, decay_factors[:,1], ts)

        return omegas, zetas

    def initialize_parameters(self, params, ts):
        omega, zeta = params

        # A = jax.vmap(lambda o, z: jnp.array([[0,1],[-o,-z]]))(omega, zeta)
        # self.A = diffrax.LinearInterpolation(ts, A)
        self.A = jnp.array([[0,1],[-omega, -zeta]])

        self.b = jnp.array([[0.0,1.0]]).T
        self.G = jnp.array([[0,0],[0,1]])
        self.V = self.process_noise*self.G

        self.C = jnp.eye(self.n_var)[:self.n_obs]
        self.W = self.obs_noise*jnp.eye(self.n_obs)

    def drift(self, t, state, args):
        # print(self.A.shape, state.shape, self.b.shape, args.shape)
        return self.A@state + self.b@args
    
    def diffusion(self, t, state, args):
        return self.V
    
    def fitness_function(self, state, control, target, ts):
        x_d = jnp.stack([jnp.squeeze(target), jnp.zeros_like(target)], axis=1)

        u_d = jax.vmap(lambda _x: -jnp.linalg.pinv(self.b)@self.A@_x)(x_d)
        costs = jax.vmap(lambda _state, _u, _x_d, _u_d: (_state-_x_d).T@self.Q@(_state-_x_d) + (_u-_u_d)@self.R@(_u-_u_d))(state,control, x_d, u_d)
        return jnp.sum(costs)
    
    def terminate_event(self, state, **kwargs):
        return jnp.any(jnp.isnan(state.y))# | jnp.any(jnp.isinf(state.y))