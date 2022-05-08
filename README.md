# Evolutionary algorithms for reinforcement learning

This repository is based on the <a href="https://github.com/OhGreat/evolutionary_algorithms">evolutionary algorithms</a> repository. It extends the framework by applying evolutionary strategies to train network agents that can solve OpenAI's gym environments. All environments available in `Classic Control`, together with some from `Box2D` have been tested and work with the available implementation. The following gifs are examples of agents trained with various configurations of this framework:

<p float="left">
  <img src="https://github.com/OhGreat/es_for_rl_experimentation/blob/main/readme_aux/cartpole_c.gif" width="320" />
  <img src="https://github.com/OhGreat/es_for_rl_experimentation/blob/main/readme_aux/lunar_lander_c.gif" width="320" /> 
  <img src="https://github.com/OhGreat/es_for_rl_experimentation/blob/main/readme_aux/acrobot_c.gif" width="320" />
</p>


## Prerequisites

In order to use this repository, a `Python 3` environment is required, with the packages specified in the `requirements.txt` file, in the main directory, which can be installed with pip.  

## Usage

The framework has two main uses, `training` and `evaluating` configurations. Training consists in using evolutionary algorithms in order to train the weights of the model network used to sample actions. Evaluating consists in letting the agent play with the environment for a set number of repetitions to collect statistics and averege results of our training.  How to use the framework for each task is explained in detail in the following subsections.

### Training a configuration

The main python file to train networks is the `train_network.py`, in the main directory. Example shell scripts are available in the `exp_scripts` directory. The following arguments can be set when running the training script: 
- `-exp_name` : defines the name of the experiment.
- `-b` : defines the total amount of evaluations.
- `-min` : use this flag if the problem is minimization.
- `-r` : defines the recombination strategy.
- `-m` : defines the mutation strategy.
- `-s` : defines the selection strategy.
- `-ps` : defines the number of parents per generation.
- `-os` : defines the number of offspring per generation.
- `-mul` : defines the multiplier for the one fifth success rule.
- `-pat` : defines the wait time before resetting sigmas.
- `-exp_reps` : defines the number of experiments to average results.
- `-train_reps` : defines the number of evaluation repetitions to use during training.
- `-eval_reps` : defines the number of evaluation simulations to run after 'training' the models.
- `-model` : defines the model architecture to use.
- `-env` : deefines the environment in which to train our agent.
- `-render_train` : used to render the training process. Only recommended for debugging purposes.
- `-render_eval` : used to render the final evaluation process after training thte model.
- `-virtual_display` : needed to run render on headless servers.
- `-plot_name` : saves plot of the training when a name for the plot is defined.
- `-env_threshold` : Optimum value to set as horizontal line in plot.
- `-v` : Defines the intensity of debug prints.

To run an example script from terminal, execute from the `main directory` the following commands, by substituting <bash_script.sh> with the name of the script you wish to use from the `exp_scripts` directory, as such: 
```
chmod +x exp_scripts/<bash_script.sh>
./exp_scripts/<bash_script.sh>
```



### Evaluating a configuration

The main python file to evaluate networks is the `eval.py`, in the main directory. Example shell scripts are also present in the `exp_scripts` directory. The following arguments can be set tot use eval.py:

- `-model` : defines the model architecture for the weights.
- `-model_weights` : path of the model weights to load for the evaluation.
- `-env` : environment the model was trained upon.
- `-eval_reps` : number of times to evaluate our individual.
- `-render_eval` : use this flag to render the evaluation process.
- `-virtual_display` : required to use render on headless servers.

The same commands for running the train scripts is also valid for running the evaluation scripts in the `exp_scripts` directory. Remember to run the scripts from the main directory, as:
```
chmod +x exp_scripts/<bash_script.sh>
./exp_scripts/<bash_script.sh>
```

## TODO

- add env step count
