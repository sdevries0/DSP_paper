import jax
import jax.numpy as jnp
from jax.tree_util import register_pytree_node
from genetic_operators.simplification import tree_to_sympy

class NetworkTrees:
    """
        Class that contains trees that represent the hidden state and readout of a model
        Attributes:
            hidden_state_trees (list[PyTree]): A list of trees for each of the neurons in the hidden state
            readout_tree (PyTree): A tree that outputs a control force
            fitness (float)
    """

    def __init__(self, trees):
        self.trees = trees
        self.fitness = None
    
    def __call__(self):
        return self.trees
    
    def set_fitness(self, fitness: float):
        self.fitness = fitness

    def reset_fitness(self):
        self.fitness = None
        
    def evaluate_tree(self, tree, expressions):
        """expression tree evaluation with variables x as input"""
        if len(tree) > 1:
            f = tree[0]
            nodes = [self.evaluate_tree(branch, expressions) for branch in tree[1:]]
            return lambda args: f(*map(lambda n: n(args), nodes))
        else:
            leaf = tree[0]
            if isinstance(leaf,jnp.ndarray):
                return lambda args: leaf
            return leaf

    def tree_to_function(self, expressions):
        functions = []
        for i in range(len(self.trees)):
            layer = self.trees[i]
            layer_functions = []
            for tree in layer:
                layer_functions.append(self.evaluate_tree(tree, expressions[i]))

            def f(layer_functions):
                return lambda args: jnp.array([func(args) for func in layer_functions])
            
            functions.append(f(layer_functions))
        return functions

    def get_params(self, values_and_paths: list):
        "Returns all optimisable parameters ##NOT USED"
        params = []
        paths = []
        for path, value in values_and_paths:
            if isinstance(value, jax.numpy.ndarray):
                params.append(value)
                paths.append(path)
        return jnp.array(params), paths
        
    def __str__(self) -> str:
        string_output = ""
        for i in range(len(self.trees)):
            string_output += "["
            layer = self.trees[i]
            for j in range(len(layer)):
                string_output += str(tree_to_sympy(layer[j]))
                if j < (len(layer) - 1):
                    string_output += ", "
            string_output += "]"
            if i < (len(self.trees) - 1):
                string_output += ", "
        return string_output

#Register the class of trees as a pytree
register_pytree_node(NetworkTrees, lambda tree: ((tree()), None),
    lambda _, args: NetworkTrees(args))

class RNN:
    def __init__(self, input_layer, recurrent, bias, action_layer, action_bias) -> None:
        self.input_layer = input_layer
        self.recurrent = recurrent
        self.bias = bias
        self.action_layer = action_layer
        self.action_bias = action_bias

    def update(self, y, a, u, target):
        x = jnp.concatenate([y, u, target])
        x = jnp.tanh(self.input_layer@x + self.recurrent@a + self.bias)
        return x
    
    def act(self, a, target):
        x = jnp.concatenate([a, target])

        return self.action_layer@x + self.action_bias
    
class ParameterReshaper:
    def __init__(self, obs_space, latent_size, action_space, n_targets):
        self.input_layer_shape = (obs_space + action_space + n_targets, latent_size)
        self.recurrent_layer_shape = (latent_size, latent_size)
        self.bias_shape = (latent_size,)

        self.action_layer_shape = (latent_size + n_targets, action_space)
        self.action_bias_shape = (action_space,)

        self.total_parameters = jnp.sum(jnp.array([*map(lambda l: l[0]*l[1], [self.input_layer_shape, self.recurrent_layer_shape, self.action_layer_shape])])) \
                            + jnp.sum(jnp.array([*map(lambda l: l[0], [self.bias_shape, self.action_bias_shape])]))

    def __call__(self, params):
        assert params.shape[0] == self.total_parameters

        index = 0
        layers = []

        w = params[index:index+self.input_layer_shape[0]*self.input_layer_shape[1]].reshape(self.input_layer_shape[1], self.input_layer_shape[0])
        index += self.input_layer_shape[0]*self.input_layer_shape[1]
        layers.append(w)

        w = params[index:index+self.recurrent_layer_shape[0]*self.recurrent_layer_shape[1]].reshape(self.recurrent_layer_shape[1], self.recurrent_layer_shape[0])
        index += self.recurrent_layer_shape[0]*self.recurrent_layer_shape[1]
        layers.append(w)

        w = params[index:index+self.bias_shape[0]]
        index += self.bias_shape[0]
        layers.append(w)

        w = params[index:index+self.action_layer_shape[0]*self.action_layer_shape[1]].reshape(self.action_layer_shape[1], self.action_layer_shape[0])
        index += self.action_layer_shape[0]*self.action_layer_shape[1]
        layers.append(w)

        w = params[index:index+self.action_bias_shape[0]]
        index += self.action_bias_shape[0]
        layers.append(w)

        return layers