from pathlib import Path
from ast import literal_eval
import numpy as np
import matplotlib.pyplot as plt
import torch


def average_repeats(root):
    root = Path(root)
    avg_metrics = {}
    avg_mxs = {}
    switch_pts = None
    count = 0
    for f in root.iterdir():
        try:
            log = torch.load(f / 'log.pt')
        except (NotADirectoryError, FileNotFoundError):
            continue
        count += 1
        switch_pts = log.switch_pts
        for name, metric in log.metrics.items():
            if name not in avg_metrics:
                avg_metrics[name] = metric
            else:
                for i in range(len(metric)):
                    avg_metrics[name][i] = (
                        avg_metrics[name][i][0],
                        avg_metrics[name][i][1] + metric[i][1]
                    )
        for name, mx in log.mxs.items():
            if name not in avg_mxs:
                avg_mxs[name] = mx
            else:
                avg_mxs[name] = (
                    avg_mxs[name][0],
                    avg_mxs[name][1] + mx[1]
                )
    for name, metric in avg_metrics.items():
        for i in range(len(metric)):
            avg_metrics[name][i] = (
                metric[i][0],
                metric[i][1] / count
            )
    for name, mx in avg_mxs.items():
        avg_mxs[name] = mx[0], mx[1] / count
    return avg_metrics, avg_mxs, switch_pts


def add_switch_pts(pts, arrows=False):
    for pt in pts:
        plt.axvline(pt,
                    color='k',
                    linestyle='--',
                    linewidth=0.8,
                    alpha=0.5)
        if arrows:
            plt.arrow(pt,
                      0.04,
                      50,
                      0,
                      head_length=25,
                      head_width=0.02,
                      alpha=0.8)


def compute_forgetting(metrics, mxs):
    forgetting = {}
    for expr_id, (batch_id, mx) in mxs.items():
        if expr_id.endswith('BL'):
            continue
        if expr_id not in forgetting:
            forgetting[expr_id[:-2] + 'BL'] = []
            forgetting[expr_id] = []
        for x, y in metrics[expr_id]:
            if x <= batch_id:
                continue
            forgetting[expr_id].append((x, mx - y))
        if len(forgetting[expr_id]) == 0:
            forgetting.pop(expr_id)

        # compute BL "forgetting" by comparing against
        # maximum MH acc, NOT maximum BL acc
        # essentially means that we want to see degradation
        # due to added classes not seen in LE and NOT due
        # to forgetting per se
        # this is different from the RWalk paper, which
        # compares against max BL and refers to it as forgetting

        for x, y in metrics[expr_id[:-2] + 'BL']:
            if x <= batch_id:
                continue
            forgetting[expr_id[:-2] + 'BL'].append((x, mx - y))
        if len(forgetting[expr_id[:-2] + 'BL']) == 0:
            forgetting.pop(expr_id[:-2] + 'BL')

    count_MH = {}
    average_MH = {}
    count_BL = {}
    average_BL = {}
    for expr_id, forg in forgetting.items():
        for x, y in forg:
            if expr_id.endswith('BL'):
                if x not in count_BL:
                    count_BL[x] = 1.0
                    average_BL[x] = y
                else:
                    count_BL[x] += 1.0
                    average_BL[x] += y
            elif expr_id.endswith('MH'):
                if x not in count_MH:
                    count_MH[x] = 1.0
                    average_MH[x] = y
                else:
                    count_MH[x] += 1.0
                    average_MH[x] += y
    for x in count_MH:
        average_MH[x] /= count_MH[x]
    forgetting['average-MH'] = tuple(average_MH.items())
    for x in count_BL:
        average_BL[x] /= count_BL[x]
    forgetting['average-BL'] = tuple(average_BL.items())
    return forgetting


def compute_forgetting2(metrics, mxs):
    forgetting = {}
    for expr_id, (batch_id, mx) in mxs.items():
        if expr_id.endswith('BL'):
            continue
        if expr_id not in forgetting:
            forgetting[expr_id] = []
        for x, y in metrics[expr_id]:
            if x <= batch_id:
                continue
            forgetting[expr_id].append((x, mx - y))
        if len(forgetting[expr_id]) == 0:
            forgetting.pop(expr_id)

    confusion = {}
    for expr_id in metrics:
        if expr_id.endswith('BL') or 'avg' in expr_id or 'SH' in expr_id:
            continue
        bl_name = expr_id[:-2] + 'BL'
        confusion[bl_name] = []
        for i, (x, y) in enumerate(metrics[expr_id]):
            y_bl = metrics[bl_name][i][1]
            confusion[bl_name].append((x, y - y_bl))

        if len(confusion[bl_name]) == 0:
            confusion.pop(bl_name)

    def avg(d):
        count = {}
        total = {}
        average = {}
        for k, v in d.items():
            for x, y in v:
                if x not in count:
                    count[x] = 1.0
                    total[x] = y
                else:
                    count[x] += 1.0
                    total[x] += y
        for x in count:
            average[x] = total[x] / count[x]
        return tuple(average.items())

    forgetting_avg = avg(forgetting)
    confusion_avg = avg(confusion)

    return forgetting, confusion, forgetting_avg, confusion_avg


def plot_fc(d, d_avg, ylabel, spts=None, save_path=None):
    _, ax = plt.subplots()
    for expr_id in d:
        ax.plot(*zip(*d[expr_id]),
                alpha=0.4)

    ax.plot(*zip(*d_avg),
            'k',
            linewidth=0.8,
            label='average')

    plt.xlabel('Batches seen')
    plt.ylabel(ylabel)
    plt.ylim([-0.1, 1.0])
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0,
                     chartBox.width*0.75, chartBox.height])
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.25, 1.0),
              shadow=True, ncol=1)

    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    final = d_avg[-1][1]
    ax.text(0.05,
            0.95,
            f'final average = {final:.4f}',
            transform=ax.transAxes,
            fontsize=11,
            verticalalignment='top',
            bbox=props)

    if spts is not None:
        add_switch_pts(spts)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def plot_forgetting(forgetting, spts=None, save_path=None):
    _, ax = plt.subplots()
    for expr_id in forgetting:
        if 'average' in expr_id:
            continue
        if 'MH' in expr_id:
            p = ax.plot(*zip(*forgetting[expr_id]),
                        alpha=0.4, marker='x', markevery=20, markersize=2)
            ax.plot(*zip(*forgetting[expr_id[:-2] + 'BL']),
                    color=p[-1].get_color(),
                    alpha=0.4, marker='v', markevery=20, markersize=2)

    ax.plot(*zip(*forgetting['average-BL']),
            'k', marker='x', markevery=20,
            markersize=3, linewidth=0.8, label='average decay')
    ax.plot(*zip(*forgetting['average-MH']),
            'k', marker='o', markevery=20,
            markersize=3, linewidth=0.8, label='average forgetting')

    plt.xlabel('Batches seen')
    plt.ylabel('Decay')
    plt.ylim([-0.1, 1.0])
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0,
                     chartBox.width*0.75, chartBox.height])
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.25, 1.0),
              shadow=True, ncol=1)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    fmh = forgetting['average-MH'][-1][1]
    fbl = forgetting['average-BL'][-1][1]
    ax.text(0.05, 0.95,
            f'average forgetting = {fmh:.4f}\naverage decay = {fbl:.4f}',
            transform=ax.transAxes, fontsize=11,
            verticalalignment='top', bbox=props)
    if spts is not None:
        add_switch_pts(spts)
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def get_baselines(metrics):
    return {i: j[-1][1] for i, j in metrics.items()}


def compute_intransigence(mxs, baseline_accuracies):
    intra = []
    for name, (batch_id, mx_acc) in mxs.items():
        if 'MH' in name and 'avg' not in name:
            intra.append(baseline_accuracies[name] - mx_acc)
    return intra


def plot_accuracy(metrics,
                  mxs,
                  spts=None,
                  b=None,
                  intransigence=None,
                  save_path=None):
    avg_baseline = 0.
    count = 0
    marked = False
    _, ax = plt.subplots()
    for name in metrics:
        if name.endswith('MH') and 'avg' not in name:
            x0 = metrics[name][0][0]
            x1 = mxs[name][0]
            if b is not None:
                avg_baseline += b[name]
                count += 1
                if marked:
                    ax.plot([x0, x1], [b[name], b[name]],
                            '.-.r', alpha=.7, linewidth=0.8, markersize=2)
                else:
                    ax.plot([x0, x1], [b[name], b[name]],
                            '.-.r', alpha=.7, linewidth=0.8, markersize=2,
                            label='baseline')
                    marked = True
            m0 = [(x, y) for x, y in metrics[name] if x <= x1]
            m1 = [(x, y) for x, y in metrics[name] if x > x1]
            p = ax.plot(*list(zip(*m0)))
            ax.plot(*list(zip(*m1)), alpha=.3, color=p[-1].get_color())
    for name, metric in metrics.items():
        if 'avg-MH' in name:
            ax.plot(*list(zip(*metric)), '--k',
                    linewidth=0.8, label='average MH')
        elif '-SH' in name:
            ax.plot(*list(zip(*metric)), '--k', marker='v',
                    markersize=3, markevery=20,
                    linewidth=0.8, label='SH')
            if b is not None:
                ax.axhline(b[name], linestyle='-.', c='g',
                           linewidth=0.8, label='baseline SH')
    if b is not None:
        plt.axhline(avg_baseline / count, linestyle='-.',
                    c='r', linewidth=0.8, label='baseline average MH')
    if spts is not None:
        add_switch_pts(spts)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0,
                     chartBox.width*0.75, chartBox.height])
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.25, 1.0),
              shadow=True, ncol=1)
    if intransigence is not None:
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, f'average intransigence = {intransigence:.4f}',
                transform=ax.transAxes, fontsize=11,
                verticalalignment='top', bbox=props)
    plt.ylim([0, 1.1])
    plt.xlabel('Batches seen')
    plt.ylabel('Accuracy')
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def bar_plot(accuracies, labels, baselines=None, save_path=None):
    width = 0.05
    n = len(accuracies)
    m = len(accuracies[0])
    x = np.arange(1, m + 1)
    c = n // 2
    _, ax = plt.subplots()
    for i, acc in enumerate(accuracies):
        ax.bar(x + (i - c) * width, acc, width=width, label=labels[i])
    if baselines is not None:
        k = 0 if n % 2 == 0 else 1
        for i, b in enumerate(baselines):
            ax.plot([i + 1 - (c + 1) * width, i + 1 +
                     (c + k) * width], [b, b], '-.r',
                    label='baselines' if i == 0 else None)
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0,
                     chartBox.width*0.75, chartBox.height])
    ax.legend(loc='upper center',
              bbox_to_anchor=(1.25, 1.0),
              shadow=True, ncol=1)
    plt.xlabel('Learning experience')
    plt.ylabel('Accuracy on current LE at the end of LE')
    plt.ylim([0, 1.1])
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()


def read_label(fpath, label_name, sublabel_name=None):
    label = None
    with open(fpath, 'r') as f:
        for line in f:
            if line.startswith(label_name):
                label = line.split(label_name)[1][2:]
    if sublabel_name is not None:
        label = literal_eval(label)
        label = label[sublabel_name]
    return label


def bar_plot_from_folders(folders,
                          label_name='model_params',
                          sublabel_name='reg_coef',
                          baseline_folder=None,
                          save_path=None):
    labels = []
    accuracies = []
    keys = None
    for f in folders:
        settings = Path(f) / 'settings.txt'
        label = read_label(settings, label_name, sublabel_name)
        if sublabel_name is None:
            labels.append(f'{label_name}: {label}')
        else:
            labels.append(f'{sublabel_name}: {label}')
        _, mxs, _ = average_repeats(f)
        if keys is None:
            keys = [k for k in mxs if 'MH' in k and 'avg' not in k]
        acc = [mxs[k][1] for k in keys]
        accuracies.append(acc)
    if baseline_folder is not None:
        bmet, _, _ = average_repeats(baseline_folder)
        baselines = get_baselines(bmet)
        baselines = [baselines[k] for k in keys]
    else:
        baselines = None

    bar_plot(accuracies, labels, baselines, save_path)


def bar_plot_from_root(root,
                       label_name='model_params',
                       sublabel_name='reg_coef',
                       baseline_folder=None,
                       save_path=None):
    root = Path(root)
    folders = list(root.iterdir())
    bar_plot_from_folders(folders,
                          label_name,
                          sublabel_name,
                          baseline_folder,
                          save_path)
