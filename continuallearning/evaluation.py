from continuallearning.utils import map_targets
from torch.utils.data import DataLoader
from collections import OrderedDict
import torch


def accuracy(y, t):
    ym = y.argmax(dim=1)
    return ym.eq(t.to(ym)).sum().float().item() / ym.shape[0]


def evaluate(model, evu, batch_size):
    current_task = model.active_layer
    metrics = OrderedDict()
    for task_name, tagged_dataset, reporting_order in evu.evu_tasks:
        loader = DataLoader(tagged_dataset, batch_size=batch_size)
        model.activate(task_name)
        output = None
        target = tuple()
        for x, t in loader:
            if torch.cuda.is_available():
                x = x.to(torch.device('cuda'))
            y = model.predict_batch(x)
            if output is None:
                output = y
            else:
                for i in y.keys():
                    output[i] = torch.cat([output[i], y[i]], dim=0)
            target += t

        all_tags = tuple([i for i in output.keys() if i != 'unknown'])
        all_t = map_targets(target, all_tags)
        all_y = torch.cat([output[i] for i in all_tags],
                          dim=1).to(torch.device('cpu'))

        metrics[f'{task_name}-SH'] = accuracy(all_y, all_t)

        metrics[f'{task_name}-avg-BL'] = 0.
        metrics[f'{task_name}-avg-MH'] = 0.
        for expr_id, expr_tags in reporting_order.items():

            # select rows (samples)
            expr_indices = torch.tensor(
                [i for i, ti in enumerate(target) if ti in expr_tags]
            )
            expr_t = all_t.index_select(0, expr_indices)
            expr_y = all_y.index_select(0, expr_indices)
            acc = accuracy(expr_y, expr_t)
            metrics[f'{task_name}-{expr_id}-BL'] = acc
            metrics[f'{task_name}-avg-BL'] += acc

            # select columns (classes), ie, make it multi-head
            expr_target = tuple([ti for ti in target if ti in expr_tags])
            expr_t = map_targets(expr_target, expr_tags)
            expr_y = torch.cat([output[i] for i in expr_tags],
                               dim=1).to(torch.device('cpu'))
            expr_y = expr_y.index_select(0, expr_indices)
            acc = accuracy(expr_y, expr_t)
            metrics[f'{task_name}-{expr_id}-MH'] = acc
            metrics[f'{task_name}-avg-MH'] += acc

        metrics[f'{task_name}-avg-BL'] /= len(reporting_order)
        metrics[f'{task_name}-avg-MH'] /= len(reporting_order)

    model.activate(current_task)
    return metrics
