# Requirements
The code contained here was tested with:

- [PyTorch](https://pytorch.org/) 1.1.0
- [Torchvision](https://pytorch.org/) 0.3.0
- [Matplotlib](https://matplotlib.org/) 3.1.2
- [Numpy](https://numpy.org/) 1.17.4
- [Scikit-learn](https://scikit-learn.org/stable/index.html) 0.22 
- [Scipy](https://www.scipy.org/) 1.4.1

To run the notebook, you will also need to install [jupyter](https://jupyter.org/).


# The datasets
We rely in our experiments on publicly available datasets, such as MNIST and CIFAR. Those datasets will be automatically downloaded by the training script.

NOTE: some of these datasets are large in size.

# The models
All the models are defined in the `continuallearning/models.py` file. As you will see, all the models inherit from the single base class that provides the skeleton for all continual learning models. It is written such that it can be easily extended to test new fixed-capacity models and ideas.

Below are the names of the models as they appear in the `continuallearning/models.py`, and how they are passed to the `train_fc` in the `train.py` script.

| Class name | `model_name` in `train.py`|
| :----------: | -----------: |
| Vanilla | vanilla |
| L2 | l2 |
|EWC|ewc|
|RWalk|rwalk|
|AverageActivation| avga|
| AverageLDA|avgl|

---


# The main training script
The main training script, and the file you should edit, is the `train.py` file. As it is set up now, it will execute the `c100-2` experiment. But you can change that in the script.

The script contains the main parameters you can change to set different experiments, and they are documented in the comments.

The script will first run a baseline model (which is need to compute intransigence), and then L2, EWC, and RWalk. The script is set to repeat each experiment 10 times and average, hence it will take significant time to finish execution.

Note that these files are written to demo results in an simple way that does not require the user to edit many lines. Hence, the code is a bit restrictive in what it expects. Each different run of the same model and episode (e.g., different regularization coefficient) should be done under a different `exp_name`. Otherwise, existing results will be overwritten.

# Generating figures
Once the `train.py` script finishes execution, it will generate logs and figures in the `logs` folder. Once that is done, you can open the `generate_plots.ipynb` notebook (this is jupyter notebook) to compare accuracy values. You can execute the cells in order. Make sure the `exp_name` and `episode_name` are set as needed (to correspond to the same values used in the `train.py` script).

# The episodes
You'll find episodes defined in the `episodes` folder. The available episodes are: `c100-2`, `diverse-2`, `c10-2`, `k-2`, `c10-k-2`, and `k-c10-2`.

This repo already contains the logs and results for the `c100-2` episode after running the `train.py` script and the `generate_plots.ipynb` notebook. To generate results for other episodes, change the `episode_name` in `train.py` to the needed episode. Save changes and execute the script. Once logs are ready, run the `generate_plots.ipynb` notebook again, and take note of the `episode_name` in the notebook as well.