import math
import torch
from torch.utils.data import DataLoader
from continuallearning.evaluation import evaluate


def learn_experience(model,
                     expr,
                     evu,
                     batch_size,
                     n_batches,
                     eval_every,
                     scheduler=None):
    loader = DataLoader(expr.dataset, batch_size=batch_size, shuffle=False)
    model.reset_training_loader(loader)

    model.activate(expr.task_name)
    batch_count = 0
    done = False
    while True:
        if done:
            break
        for x, t in loader:
            if torch.cuda.is_available():
                x = x.to(torch.device('cuda'))
            loss = model.fit_batch(x,
                                   t,
                                   expr.unit_names,
                                   expr.loss_fn,
                                   pos_weight=expr.pos_weight)
            batch_count += 1
            if scheduler is not None:
                scheduler.step()
            print(f'batch {batch_count}/{n_batches}')
            print(f'{expr.task_name}-{expr.id} training loss: {loss:.4f}')
            model.logger.log(loss=loss)
            if batch_count % eval_every == 0:
                metrics = evaluate(model, evu, batch_size)
                print('-----------------\n')
                print(f'evaluation accuracy')
                for metric_id, metric in metrics.items():
                    print(f'{metric_id}: {metric:.4f}')
                    model.logger.log(task_name=metric_id, metric=metric)
                print('-----------------\n')

                # log max at end of learning experience for current expr-id
                # used for computing forgetting later
                try:
                    # raises KeyError for baselines!
                    model.logger.log(
                        task_name=f'{expr.task_name}-{expr.id}-MH',
                        metric=metrics[f'{expr.task_name}-{expr.id}-MH'],
                        mx=True)
                    model.logger.log(
                        task_name=f'{expr.task_name}-{expr.id}-BL',
                        metric=metrics[f'{expr.task_name}-{expr.id}-BL'],
                        mx=True)
                except KeyError:
                    pass
            if batch_count == n_batches:
                done = True
                break


def learn_episode(model,
                  episode,
                  opt,
                  lr1,
                  lr2,
                  batch_size,
                  n_batches_frozen,
                  n_batches_tuning,
                  eval_every,
                  adapt_lr=False,
                  fct1=1.0,
                  mid_learn=False,
                  mid_learn_n_batches=0):
    if torch.cuda.is_available():
        model.to(torch.device('cuda'))

    for epii, (expr, evu) in enumerate(episode):

        model.pre_learning(expr)

        model.extend_output_layer(expr.task_name, expr.unit_names)
        model.logger.log(switch_pt=True)

        if n_batches_frozen > 0:
            # freeze encoder and learn new units
            trainable_params_1, trainable_params_2 = freeze_except(
                model, expr.task_name, expr.unit_names, freeze_encoder=True)
            model.init_optimizer(opt, trainable_params_1, trainable_params_2,
                                 lr1, lr2)
            learn_experience(model,
                             expr,
                             evu,
                             batch_size,
                             n_batches_frozen,
                             eval_every,
                             scheduler=None)

        # fine-tune everything
        trainable_params_1, trainable_params_2 = freeze_except(
            model, expr.task_name, expr.unit_names, freeze_encoder=False)
        model.init_optimizer(opt, trainable_params_1, trainable_params_2, lr1,
                             lr2)
        if adapt_lr:

            def lambda1(bst):
                return 1 - math.exp(-bst / n_batches_tuning * fct1)

            def lambda2(bst):
                return 1.0

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                model.optimizer, lr_lambda=[lambda1, lambda2])
        else:
            scheduler = None

        learn_experience(model,
                         expr,
                         evu,
                         batch_size,
                         n_batches_tuning,
                         eval_every,
                         scheduler=scheduler)

        if mid_learn:
            model.mid_learning(expr)
            learn_experience(model,
                             expr,
                             evu,
                             batch_size,
                             mid_learn_n_batches,
                             eval_every,
                             scheduler=scheduler)

        model.post_learning(expr)


def freeze_except(model, layer_names, unit_names, freeze_encoder=False):
    trainable_params_1 = []
    trainable_params_2 = []
    for param in model.encoder.parameters():
        if freeze_encoder:
            param.requires_grad = False
        else:
            param.requires_grad = True
            trainable_params_1.append(param)

    if hasattr(model, 'decoder'):
        for param in model.decoder.parameters():
            if freeze_encoder:
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params_1.append(param)

    if hasattr(model, 'temp_heads'):
        for param in model.temp_heads.parameters():
            if freeze_encoder:
                param.requires_grad = False
            else:
                param.requires_grad = True
                trainable_params_1.append(param)

    # always optimize alternate heads, even with frozen
    if hasattr(model, 'alternate_heads'):
        for param in model.alternate_heads.parameters():
            param.requires_grad = True
            trainable_params_1.append(param)

    for layer_name, layer in model.output_layers.items():
        for unit_name, unit in layer.output_units.items():
            for param in unit.parameters():
                if layer_name in layer_names and unit_name in unit_names:
                    param.requires_grad = True
                    trainable_params_2.append(param)
                else:
                    param.requires_grad = False
    return trainable_params_1, trainable_params_2
