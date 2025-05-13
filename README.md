# Discovering Dynamic Symbolic Policies with Genetic Programming

In this repository, you can find the source code used for the paper "Discovering Dynamic Symbolic Policies with Genetic Programming". Click [here](https://arxiv.org/abs/2406.02765) to read the paper. 
In this paper, we used genetic programming to evolve control policies consisting of symbolic expressions, both with and without memory. 
The code makes use of the [Kozax](https://github.com/sdevries0/Kozax) framework, which runs genetic programming in JAX.

## Build
To use the code, you can clone the repository and create the environment by running:
```
conda env create -f environment.yml
conda activate gp_policies
```

In `run.py`, you can select the algorithm, environment and setting to use in an experiment, as well as change the hyperparameter settings of the evolutionary algorithms. 'lag_experiment.py' contains the code run the comparison between the dynamic policies and static policies with time-lagged observations. 'generate_plots_ipynb' provides the code for recreating the figures in the paper.

## Citation
If you make use of this code in your research paper, please cite:
```
@article{de2024discovering,
  title={Discovering Dynamic Symbolic Policies with Genetic Programming},
  author={de Vries, Sigur and Keemink, Sander and van Gerven, Marcel},
  journal={arXiv preprint arXiv:2406.02765},
  year={2024}
}
```

