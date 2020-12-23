from pathlib import Path
import matplotlib.pyplot as plt
from plot import compute_forgetting2, plot_fc
from plot import get_baselines, compute_intransigence
from plot import plot_accuracy, average_repeats
import torch
from continuallearning.learning import learn_episode
from continuallearning.models import Vanilla, AntReg, L2, EWC, Replay
from continuallearning.models import LwF, AverageActivation, AverageLDA
from continuallearning.models import ReplayEWC, ReplayLwF, RWalk
from continuallearning.load_data import Episode, create_baseline_episode
from continuallearning.cores import Encoder, Decoder


def train_fc(episode_name,
             model_name,
             model_params,
             baseline,
             n_batches_tuning,
             dev,
             repeats,
             activation,
             batch_size,
             eval_every,
             exp_name):

    # Location of data.
    data_root = 'data'

    # Number of units in the last layer before the output layer
    # (ie the second fully connected layer).
    n_penultimate = 256

    # Number of units in the first fully connected layer.
    fc1nb = 1024

    # Number of conv units in each conv layer.
    cnb = 128

    adapt_lr = False
    fct1 = 1.0

    cores = 'basic'
    n_batches_frozen = 0

    fc_dropout = None
    conv_dropout = None
    batchnorm = True

    mid_learn = False
    mid_learn_n_batches = 0

    test_samples_per_class = None
    train_samples_per_class = 100

    aux_samples_per_class = None
    pos_weight = 1.0
    loss_fn = 'multi'
    opt = 'adam'
    lr1 = 1e-4
    lr2 = 1e-4
    # The shape of the expected input. The models expect this shape.
    # If your data is in a different shape, reshape it first.
    shape = 3, 32, 32

    episode_config = 'episodes/' + episode_name + '.json'

    logs_folder = Path('logs').expanduser()
    if baseline:
        log_folder = logs_folder / exp_name / 'baselines' / episode_name
    else:
        log_folder = logs_folder / exp_name / model_name / \
            episode_name / 'main'
    log_folder.mkdir(parents=True)

    with (log_folder / 'settings.txt').open('a') as f:
        f.write(f'loss_fn: {loss_fn}\n')
        f.write(f'dev: {dev}\n')
        f.write(f'test_samples_per_class: {test_samples_per_class}\n')
        f.write(f'train_samples_per_class: {train_samples_per_class}\n')
        f.write(f'aux_samples_per_class: {aux_samples_per_class}\n')
        f.write(f'episode_config: {episode_config}\n')
        f.write(f'model_name: {model_name}\n')
        f.write(f'model_params: {model_params}\n')
        f.write(f'opt: {opt}\n')
        f.write(f'lr1: {lr1}\n')
        f.write(f'lr2: {lr2}\n')
        f.write(f'fct1: {fct1}\n')
        f.write(f'adapt_lr: {adapt_lr}\n')
        f.write(f'pos_weight: {pos_weight}\n')
        f.write(f'batch_size: {batch_size}\n')
        f.write(f'n_batches_frozen: {n_batches_frozen}\n')
        f.write(f'n_batches_tuning: {n_batches_tuning}\n')
        f.write(f'eval_every: {eval_every}\n')
        f.write(f'shape: {shape}\n')
        f.write(f'n_penultimate: {n_penultimate}\n')
        f.write(f'cores: {cores}\n')
        f.write(f'batchnorm: {batchnorm}\n')
        f.write(f'fc_dropout: {fc_dropout}\n')
        f.write(f'conv_dropout: {conv_dropout}\n')
        f.write(f'activation: {activation}\n')
        f.write(f'fc1nb: {fc1nb}\n')
        f.write(f'cnb: {cnb}\n')
        f.write(f'mid_learn: {mid_learn}\n')
        f.write(f'mid_learn_n_batches: {mid_learn_n_batches}\n')

    # load baseline results to compare against later
    baseline_accuracies = None
    if not baseline:
        try:
            if episode_name.endswith('-aux'):
                epn = episode_name[:-4]
            else:
                epn = episode_name
            baseline_metrics, _, _ = average_repeats(logs_folder / exp_name /
                                                     'baselines' / epn)
            baseline_accuracies = get_baselines(baseline_metrics)
        except (NotADirectoryError, FileNotFoundError):
            pass
        if not baseline_accuracies:  # if returned empty dict
            baseline_accuracies = None

    for i in range(repeats):
        rep_folder = log_folder / str(i)
        rep_folder.mkdir()

        if baseline:
            episode = create_baseline_episode(
                episode_config=episode_config,
                dev=dev,
                train_samples_per_class=train_samples_per_class,
                test_samples_per_class=test_samples_per_class,
                root=data_root)
        else:
            episode = Episode.generate_from_json(
                episode_config,
                loss_fn,
                dev,
                train_samples_per_class=train_samples_per_class,
                test_samples_per_class=test_samples_per_class,
                aux_samples_per_class=aux_samples_per_class,
                pos_weight=pos_weight,
                root=data_root)

        if cores == 'basic':
            encoder = Encoder(in_shape=shape,
                              out_features=n_penultimate,
                              batchnorm=batchnorm,
                              activation=activation,
                              fc_dropout=fc_dropout,
                              conv_dropout=conv_dropout,
                              fc1nb=fc1nb,
                              cnb=cnb)
            decoder = Decoder(in_features=n_penultimate,
                              out_shape=shape,
                              batchnorm=batchnorm,
                              fc1nb=fc1nb,
                              cnb=cnb)
        else:
            raise Exception('unrecognized cores')

        if model_name == 'vanilla':
            model = Vanilla(encoder, n_penultimate)
        elif model_name == 'antreg':
            model = AntReg(encoder,
                           n_penultimate,
                           decoder,
                           reg_coef=model_params['reg_coef'],
                           rec_loss_fn=model_params['rec_loss_fn'],
                           rand_cutoff=model_params['rand_cutoff'])
        elif model_name == 'l2':
            model = L2(encoder,
                       n_penultimate,
                       reg_coef=model_params['reg_coef'],
                       all_anchors=model_params['all_anchors'],
                       clip_gradients=model_params['clip_gradients'])
        elif model_name == 'avga':
            model = AverageActivation(
                encoder,
                n_penultimate,
                reg_coef=model_params['reg_coef'],
                reg_coef_deeply=model_params['reg_coef_deeply'],
                alpha=model_params['alpha'],
                all_anchors=model_params['all_anchors'],
                cnb=cnb,
                fc1nb=fc1nb
            )
        elif model_name == 'avgl':
            model = AverageLDA(
                encoder,
                n_penultimate,
                reg_coef=model_params['reg_coef'],
                reg_coef_avg=model_params['reg_coef_avg'],
                reg_coef_deeply=model_params['reg_coef_deeply'],
                alpha=model_params['alpha'],
                all_anchors=model_params['all_anchors'],
                cnb=cnb,
                fc1nb=fc1nb
            )
        elif model_name == 'ewc':
            model = EWC(encoder,
                        n_penultimate,
                        reg_coef=model_params['reg_coef'],
                        all_anchors=model_params['all_anchors'],
                        fisher_samples=model_params['fisher_samples'],
                        empirical_fisher=model_params['empirical_fisher'],
                        simple_fisher=model_params['simple_fisher'],
                        fisher_loss_fn=model_params['fisher_loss_fn'],
                        clip_gradients=model_params['clip_gradients'],
                        include_unknown=model_params['include_unknown'])
        elif model_name == 'rwalk':
            model = RWalk(encoder,
                          n_penultimate,
                          reg_coef=model_params['reg_coef'],
                          fisher_alpha=model_params['fisher_alpha'],
                          clip_gradients=model_params['clip_gradients'])
        elif model_name == 'lwf':
            model = LwF(encoder,
                        n_penultimate,
                        reg_coef=model_params['reg_coef'],
                        batch_size=batch_size)
        elif model_name == 'replay':
            model = Replay(encoder,
                           n_penultimate,
                           memory_budget=model_params['memory_budget'],
                           batch_size=batch_size,
                           replay_period=model_params['replay_period'],
                           replay_epochs=model_params['replay_epochs'])
        elif model_name == 'replay_lwf':
            model = ReplayLwF(encoder,
                              n_penultimate,
                              reg_coef=model_params['reg_coef'],
                              batch_size=batch_size,
                              memory_budget=model_params['memory_budget'],
                              replay_period=model_params['replay_period'],
                              replay_epochs=model_params['replay_epochs'],
                              scale_replay=model_params['scale_replay'],
                              scalec=model_params['scalec'])
        elif model_name == 'replay_ewc':
            model = ReplayEWC(encoder,
                              n_penultimate,
                              reg_coef=model_params['reg_coef'],
                              batch_size=batch_size,
                              memory_budget=model_params['memory_budget'],
                              replay_period=model_params['replay_period'],
                              replay_epochs=model_params['replay_epochs'],
                              scale_replay=model_params['scale_replay'],
                              scalec=model_params['scalec'],
                              all_anchors=model_params['all_anchors'],
                              fisher_samples=model_params['fisher_samples'])

        else:
            raise Exception('unrecognized model_name')

        learn_episode(model,
                      episode,
                      opt,
                      lr1,
                      lr2,
                      batch_size,
                      n_batches_frozen,
                      n_batches_tuning,
                      eval_every,
                      adapt_lr,
                      fct1,
                      mid_learn,
                      mid_learn_n_batches)

        torch.save(model.logger, rep_folder / 'log.pt')

        if not baseline:
            plot_accuracy(model.logger.metrics,
                          model.logger.mxs,
                          spts=model.logger.switch_pts,
                          b=baseline_accuracies,
                          save_path=rep_folder / 'accuracy.pdf')

            f, c, favg, cavg = compute_forgetting2(model.logger.metrics,
                                                   model.logger.mxs)
            plot_fc(f,
                    favg,
                    'forgetting',
                    model.logger.switch_pts,
                    save_path=rep_folder / 'forgetting.pdf')
            plot_fc(c,
                    cavg,
                    'confusion',
                    model.logger.switch_pts,
                    save_path=rep_folder / 'confusion.pdf')
        else:
            _, ax = plt.subplots()
            for name, met in model.logger.metrics.items():
                if 'avg' in name:
                    ax.plot(*zip(*met), '--k', label=name)
                else:
                    ax.plot(*zip(*met), label=name)
            plt.legend()
            plt.ylim([0, 1])
            plt.savefig(rep_folder / 'accuracy.pdf')

        plt.close('all')

    metrics, mxs, spts = average_repeats(log_folder)
    if not baseline:
        f, c, favg, cavg = compute_forgetting2(metrics, mxs)

        intrasigence = compute_intransigence(mxs, baseline_accuracies)
        intrasigence = sum(intrasigence) / len(intrasigence)
        plot_fc(f,
                favg,
                'forgetting',
                model.logger.switch_pts,
                save_path=log_folder / 'forgetting.pdf')
        plot_fc(c,
                cavg,
                'confusion',
                model.logger.switch_pts,
                save_path=log_folder / 'confusion.pdf')
        plot_accuracy(metrics,
                      mxs,
                      spts=spts,
                      b=baseline_accuracies,
                      intransigence=intrasigence,
                      save_path=log_folder / 'accuracy.pdf')
    else:
        _, ax = plt.subplots()
        for name, met in metrics.items():
            if 'avg' in name:
                ax.plot(*zip(*met), '--k', label=name)
            else:
                ax.plot(*zip(*met), label=name)
        plt.legend()
        plt.ylim([0, 1])
        plt.savefig(log_folder / 'accuracy.pdf')
