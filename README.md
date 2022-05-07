# Evolutionary strategies for Reinforcement learning experimentation

This repository is based on the <a href="https://github.com/OhGreat/evolutionary_algorithms">evolutionary algorithms</a> repository. It extends the framework by applying evolutionary strategies to train network agents that can solve OpenAI's gym environments. All environments available in `Classic Control` and `Box2D` have been tested and models are trainable with the EA available. The following gifs are examples of agents trained with various configurations of this framework:

<img>

## Prerequisites

In order to use this repository, a `Python 3` environment is required, with the packages specified in the `requirements.txt` file, in the main directory.  

## Usage

The framework has two main uses, training and evaluating. How to use the framework for each task is explained in detail in the following subsections.

### Training a configuration

The main python file to train networks is the `train_network.py`, in the main directory. Example shell scripts are available in the `exp_scripts` directory. The following arguments can be set when running the training script: 
- `exp_name` : defines the name of the experiment.
- `b` : defines the total amount of evaluations.
- `min` : use this flag if the problem is minimization.
- `r` : defines the recombination strategy.
- `m` : defines the mutation strategy.
- `s` : defines the selection strategy.
- `ps` : defines the number of parents per generation.
- `os` : defines the number of offspring per generation.
- `mul` : defines the multiplier for the one fifth success rule.
- `pat` : defines the wait time before resetting sigmas.
- `exp_reps` : defines the number of experiments to average results.
- `train_reps` : defines the number of evaluation repetitions to use during training.
- `eval_reps` : defines the number of evaluation simulations to run after 'training' the models.
- `model` : defines the model architecture to use.
- `env` : deefines the environment in which to train our agent.
- `render_train` : used to render the training process. Only recommended for debugging purposes.
- `render_eval` : used to render the final evaluation process after training thte model.
- `virtual_display` : needed to run render on headless servers.
- `plot_name` : saves plot of the training when a name for the plot is defined.
- `env_threshold` : Optimum value to set as horizontal line in plot.
- `v` : Defines the intensity of debug prints.

To run an example script from terminal, execute from the `main directory` the following commands, by substituting <bash_script.sh> with the name of the script you wish to use from the `exp_scripts` directory, as such: 
```
chmod +x exp_scripts/<bash_script.sh>
./exp_scripts/<bash_script.sh>
```



### Evaluating a configuration


## Examples


## TODO

- add env step count