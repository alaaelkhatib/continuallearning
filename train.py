from train_func import train_fc

# Give your experiment a name.
# Each experiment should start with baselines.
# There should be only one baseline folder per episode.
exp_name = 'demo'

# The name of the episode.
# Must be one of the episodes in the episodes folder (without the '.json')
episode_name = 'c100-2'

# How many iterations to train for per task
n_batches = 200

# Controls the number of trials to use to report averages.
# Higher values take longer to execute, but return more accurate results.
repeats = 10

# If False, reports on test set. If True, reports on dev sets.
dev = False

# The activation function to use for all layers.
# Valid values are 'leaky' for leaky ReLU or 'relu' for standard ReLU.
activation = 'relu'

# Training batch size
batch_size = 100

# Number of batches (iteration) between reported evaluations
eval_every = 100

# Run experiemnts, start with baseline if not already available

# Baseline experiment, used to report intransigence values
train_fc(
    episode_name=episode_name,
    model_name='vanilla',
    model_params={},
    baseline=True,
    n_batches_tuning=n_batches,
    dev=dev,
    repeats=repeats,
    activation=activation,
    batch_size=batch_size,
    eval_every=eval_every,
    exp_name=exp_name
)

# Vanilla model
train_fc(
    episode_name=episode_name,
    model_name='vanilla',
    model_params={},
    baseline=False,
    n_batches_tuning=n_batches,
    dev=dev,
    repeats=repeats,
    activation=activation,
    batch_size=batch_size,
    eval_every=eval_every,
    exp_name=exp_name
)

# EWC model
train_fc(
    episode_name=episode_name,
    model_name='ewc',
    model_params={'reg_coef': 1e7,
                  'all_anchors': False,
                  'fisher_samples': 1000,
                  'empirical_fisher': True,
                  'simple_fisher': True,
                  'fisher_loss_fn': None,
                  'clip_gradients': False,
                  'include_unknown': False},
    baseline=False,
    n_batches_tuning=n_batches,
    dev=dev,
    repeats=repeats,
    activation=activation,
    batch_size=batch_size,
    eval_every=eval_every,
    exp_name=exp_name
)

# RWalk model
train_fc(
    episode_name=episode_name,
    model_name='rwalk',
    model_params={'reg_coef': 1.0,
                  'fisher_alpha': 0.9,
                  'clip_gradients': False},
    baseline=False,
    n_batches_tuning=n_batches,
    dev=dev,
    repeats=repeats,
    activation=activation,
    batch_size=batch_size,
    eval_every=eval_every,
    exp_name=exp_name
)
