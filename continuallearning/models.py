import random
from copy import deepcopy
from collections import OrderedDict
import torch
from torch.optim import SGD, Adam
from torch.utils.data import DataLoader
from torch.nn import Module, ModuleDict, Linear
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.functional import softmax, log_softmax, logsigmoid, cross_entropy
from continuallearning.load_data import TaggedDataset
from continuallearning.utils import dict_binary_entropy, dict_cross_entropy
from continuallearning.utils import distillation_categorical, map_targets
import numpy as np


class Logger(object):
    def __init__(self):
        self.batches_seen = 0
        self.loss = []
        self.metrics = {}
        self.mxs = {}
        self.switch_pts = []
        self.importance = None
        self.importance_norm = None

    def log(self,
            task_name=None,
            loss=None,
            metric=None,
            switch_pt=False,
            mx=False,
            importance=None,
            importance_norm=None):
        if metric is not None:
            if task_name is None:
                raise Exception('need task_name to log metrics')
            if mx:
                self.mxs[task_name] = (self.batches_seen, metric)
            else:
                if task_name not in self.metrics:
                    self.metrics[task_name] = []
                self.metrics[task_name].append((self.batches_seen, metric))
        if loss is not None:
            self.loss.append((self.batches_seen, loss))
        if switch_pt:
            self.switch_pts.append(self.batches_seen)
        if importance is not None:
            self.importance = importance
        if importance_norm is not None:
            self.importance_norm = importance_norm

    def increment_batches_seen(self):
        self.batches_seen += 1


class OutputLayer(Module):
    def __init__(self):
        super(OutputLayer, self).__init__()
        self.output_units = ModuleDict()

    def extend(self, unit_name, unit):
        if unit_name in self.output_units:
            raise Exception('output_unit exists')
        self.output_units.update([(unit_name, unit)])

    def forward(self, z, unit_names=None):
        if unit_names is None:
            unit_names = [name for name in self.output_units]
        output = OrderedDict([(name, self.output_units[name](z))
                              for name in unit_names])
        return output


class ContinualLearningModel(Module):
    def __init__(self, encoder, n_penultimate):
        super(ContinualLearningModel, self).__init__()
        self.encoder = encoder
        self.n_penultimate = n_penultimate
        self.output_layers = ModuleDict()
        self.active_layer = None
        self.logger = Logger()
        self.optimizer = None
        self._stored_parameters = None

    def new_output_layer(self, layer_name):
        if self.contains(layer_name):
            raise Exception('layer_name exists')
        self.output_layers.update([(layer_name, OutputLayer())])

    def extend_output_layer(self, layer_name, unit_names):
        if not self.contains(layer_name):
            self.new_output_layer(layer_name)
        for unit_name in unit_names:
            if not self.contains(layer_name, unit_name):
                unit = Linear(self.n_penultimate, 1)
                unit.to(self.get_current_device())
                self.output_layers[layer_name].extend(unit_name, unit)

    def contains(self, layer_name, unit_name=None):
        if unit_name is not None:
            return unit_name in self.output_layers[layer_name].output_units
        else:
            return layer_name in self.output_layers

    def activate(self, layer_name):
        self.active_layer = layer_name

    def init_optimizer(self, opt, params1, params2, lr1, lr2):
        params = [{
            'params': params1,
            'lr': lr1
        }, {
            'params': params2,
            'lr': lr2
        }]
        if opt == 'adam':
            self.optimizer = Adam(params)
        elif opt == 'sgd':
            self.optimizer = SGD(params, momentum=0.9)
        else:
            raise Exception('unrecognized optimizer')

    def get_current_device(self):
        return next(self.parameters()).device

    def forward(self, x, unit_names=None):
        z = self.encoder(x)
        y = self.output_layers[self.active_layer](z, unit_names)
        return {'y': y}

    def predict_batch(self, x, unit_names=None):
        self.eval()
        with torch.no_grad():
            return self(x, unit_names)['y']

    @staticmethod
    def compute_loss(y, t, unit_names, loss_fn, pos_weight):
        if loss_fn == 'multi':
            return dict_cross_entropy(y, t, unit_names)
        elif loss_fn == 'binary':
            return dict_binary_entropy(y, t, unit_names, pos_weight)
        else:
            raise Exception('unrecognized loss_fn')

    def pre_learning(self, expr):
        pass

    def post_learning(self, expr):
        pass

    def mid_learning(self, expr):
        pass

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        raise NotImplementedError

    def reset_training_loader(self, loader):
        # used only for LwF
        pass


class Vanilla(ContinualLearningModel):
    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        y = self(x, unit_names=unit_names)['y']
        loss = self.compute_loss(y, t, unit_names, loss_fn, pos_weight)
        loss = sum(loss)
        loss.backward()
        self.optimizer.step()
        self.logger.increment_batches_seen()
        return loss.item()


class AntReg(ContinualLearningModel):
    def __init__(self,
                 encoder,
                 n_penultimate,
                 decoder,
                 reg_coef,
                 rec_loss_fn='binary',
                 rand_cutoff=-1):
        super(AntReg, self).__init__(encoder, n_penultimate)
        self.decoder = decoder
        self.reg_coef = reg_coef
        self.rec_loss_fn = rec_loss_fn
        self.learned = 0
        self.rand_cutoff = rand_cutoff

    def forward(self, x, unit_names=None):
        z = self.encoder(x)
        y = self.output_layers[self.active_layer](z, unit_names)
        if self.training:
            r = self.decoder(z)
            return {'y': y, 'r': r}
        else:
            return {'y': y}

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        output = self(x, unit_names)
        loss = self.compute_loss(output['y'], t, unit_names, loss_fn,
                                 pos_weight)
        loss = sum(loss)
        if random.random() > self.rand_cutoff:
            loss += self.reg_coef * self.reg_loss(output['r'], x)
        loss.backward()
        self.optimizer.step()
        self.logger.increment_batches_seen()
        return loss.item()

    def reg_loss(self, r, x):
        if self.rec_loss_fn == 'mse':
            return mse_loss(r, x)
        elif self.rec_loss_fn == 'binary':
            if x.min().item() < 0 or x.max().item() > 1:
                raise Exception('expected input normalized to [0, 1]')
            return binary_cross_entropy_with_logits(r, x)
        else:
            raise Exception('unrecognized rec_loss_fn')


class L2(ContinualLearningModel):
    def __init__(self,
                 encoder,
                 n_penultimate,
                 reg_coef,
                 all_anchors,
                 clip_gradients=True):
        super(L2, self).__init__(encoder, n_penultimate)
        self.reg_coef = reg_coef
        self.all_anchors = all_anchors
        self.reg_params = []
        self.clip_gradients = clip_gradients

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        output = self(x, unit_names)
        loss = self.compute_loss(output['y'], t, unit_names, loss_fn,
                                 pos_weight)
        loss = sum(loss)
        loss += self.reg_coef * self.reg_loss()
        loss.backward()
        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.parameters(),
                                           0.1)
        self.optimizer.step()
        self.logger.increment_batches_seen()
        return loss.item()

    def post_learning(self, expr):
        anchors = self.get_anchors()
        if self.all_anchors:
            self.reg_params.append(anchors)
        else:
            self.reg_params = [anchors]

    def get_anchors(self):
        return {
            n: deepcopy(p.detach())
            for n, p in self.encoder.named_parameters()
        }

    def reg_loss(self):
        loss = []
        for anchors in self.reg_params:
            for n, p in self.encoder.named_parameters():
                loss.append(mse_loss(p, anchors[n], reduction='sum'))
        return sum(loss)


class LDAImportance(ContinualLearningModel):
    def __init__(self,
                 encoder,
                 n_penultimate,
                 reg_coef,
                 reg_coef_deeply,
                 alpha,
                 all_anchors=False,
                 cnb=128,
                 fc1nb=1024):
        super(LDAImportance, self).__init__(encoder, n_penultimate)
        self.reg_coef = reg_coef
        self.reg_coef_deeply = reg_coef_deeply
        self.reg_params = []
        self.all_anchors = all_anchors
        self.alpha = alpha
        self.imp_log = []
        self.temp_heads = None
        self.cnb = cnb
        self.fc1nb = fc1nb

    def forward(self, x, unit_names=None):
        c1o = self.encoder.conv.conv1(x)
        c2o = self.encoder.conv.conv2(c1o)
        c3o = self.encoder.conv.conv3(c2o)
        f1o = self.encoder.mlp.fc1(self.encoder.flatten(c3o))
        f2o = self.encoder.mlp.fc2(f1o)

        y = self.output_layers[self.active_layer](
            f2o, unit_names)

        return {'y': y, 'hs': {'c1': c1o,
                               'c2': c2o,
                               'c3': c3o,
                               'f1': f1o,
                               'f2': f2o}}

    def get_anchors(self):
        return {
            n: deepcopy(p.detach())
            for n, p in self.encoder.named_parameters()
        }

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        output = self(x, unit_names)
        loss = self.compute_loss(output['y'], t, unit_names, loss_fn,
                                 pos_weight)
        loss = sum(loss)
        loss += self.reg_coef_deeply * self.deeply_loss(
            output['hs'], t, unit_names
        )
        loss += self.reg_coef * self.reg_loss()
        loss.backward()

        self.optimizer.step()
        self.logger.increment_batches_seen()
        return loss.item()

    def pre_learning(self, expr):
        device = self.get_current_device()
        self.temp_heads = ModuleDict()
        for key in ['c1', 'c2', 'c3', 'f1']:
            in_units = self.fc1nb if key == 'f1' else self.cnb
            self.temp_heads.update({
                key: torch.nn.Linear(
                    in_units,
                    len(expr.unit_names)
                ).to(device)
            })

    def deeply_loss(self, hs, t, unit_names):
        dloss = 0.
        for key in hs:
            if key == 'f2':
                continue
            if 'c' in key:
                h = hs[key].mean(dim=(2, 3))
            else:
                h = hs[key]
            yh = self.temp_heads[key](h)
            tm = map_targets(t, unit_names).to(yh.device)
            dloss += cross_entropy(yh, tm)
        return dloss

    def post_learning(self, expr):
        self.eval()
        tagged_subsets = self.split_tagged_dataset(expr.dataset,
                                                   expr.unit_names)

        # handle more than 2 classes
        assert len(tagged_subsets) == 2
        class_means = {'c1': [],
                       'c2': [],
                       'c3': [],
                       'f1': [],
                       'f2': []}
        class_vars = {'c1': [],
                      'c2': [],
                      'c3': [],
                      'f1': [],
                      'f2': []}
        imp = {'c1': [],
               'c2': [],
               'c3': [],
               'f1': [],
               'f2': []}
        for subset in tagged_subsets:
            hs = {'c1': [],
                  'c2': [],
                  'c3': [],
                  'f1': [],
                  'f2': []}
            loader = DataLoader(subset, batch_size=100, shuffle=False)
            for x, t in loader:
                if torch.cuda.is_available():
                    x = x.to(torch.device('cuda'))
                output = self(x)
                for key in hs:
                    hs[key].append(output['hs'][key].detach())

            for key in hs:
                hs[key] = torch.cat(hs[key], dim=0)
                # deal with CONV output dimensions
                if 'c' in key:
                    hs[key] = hs[key].mean(dim=(2, 3))
                class_means[key].append(hs[key].mean(dim=0))
                class_vars[key].append(hs[key].var(dim=0))

        for key in imp:
            imp[key] = self.compute_2class_imp(
                class_means[key], class_vars[key])

        self.imp_log.append({n: p.to('cpu') for n, p in imp.items()})
        self.logger.log(importance=self.imp_log)

        imp_brd = self.broadcast_imp(imp)

        anchors = self.get_anchors()
        if self.all_anchors:
            self.reg_params.append((anchors, imp_brd))
        else:
            if len(self.reg_params) > 0:
                prev_imp_brd = self.reg_params[0][1]
                for k in imp_brd:
                    # imp_brd[k] = torch.max(imp_brd[k], prev_imp_brd[k])
                    imp_brd[k] += prev_imp_brd[k]
                    imp_brd[k] *= self.alpha
            self.reg_params = [(anchors, imp_brd)]

        # USE Importance params to drive sparsity in
        # 2nd phase of learning
        # delaying the sparsity penalty relieves
        # the optimization from balancing trade-off
        # self.train()

    def reg_loss(self):
        loss = []
        for anchors, imps in self.reg_params:
            for n, p in self.encoder.named_parameters():
                loss.append((imps[n] / (imps[n].max() + 1e-9)
                             * ((p - anchors[n])**2)).sum())
        return sum(loss)

    @staticmethod
    def compute_2class_imp(class_means, class_vars):
        return (class_means[1] - class_means[0]) ** 2 / (
            sum(class_vars) + 1e-9
        )

    @staticmethod
    def split_tagged_dataset(dataset, unit_names):
        tagged_subsets = []
        for unit_name in unit_names:
            i = [i for i, t in enumerate(
                dataset.tagged_targets) if t == unit_name]
            if len(i) == 0:
                continue
            inputs_subset = torch.index_select(
                dataset.inputs, dim=0, index=torch.tensor(i))
            tagged_targets_subset = (unit_name,) * len(i)
            tagged_subsets.append(TaggedDataset(
                inputs_subset, tagged_targets_subset))
        return tagged_subsets

    @staticmethod
    def broadcast_imp(imp):
        imp_brd = {}
        imp_brd['conv.conv1.conv.weight'] = imp['c1'].reshape(-1, 1, 1, 1)
        imp_brd['conv.conv1.conv.bias'] = imp['c1']
        imp_brd['conv.conv1.batchnorm.weight'] = imp['c1']
        imp_brd['conv.conv1.batchnorm.bias'] = imp['c1']

        imp_brd['conv.conv2.conv.weight'] = imp['c2'].reshape(-1, 1, 1, 1)
        imp_brd['conv.conv2.conv.bias'] = imp['c2']
        imp_brd['conv.conv2.batchnorm.weight'] = imp['c2']
        imp_brd['conv.conv2.batchnorm.bias'] = imp['c2']

        imp_brd['conv.conv3.conv.weight'] = imp['c3'].reshape(-1, 1, 1, 1)
        imp_brd['conv.conv3.conv.bias'] = imp['c3']
        imp_brd['conv.conv3.batchnorm.weight'] = imp['c3']
        imp_brd['conv.conv3.batchnorm.bias'] = imp['c3']

        imp_brd['mlp.fc1.fc.weight'] = imp['f1'].reshape(-1, 1)
        imp_brd['mlp.fc1.fc.bias'] = imp['f1']

        imp_brd['mlp.fc2.fc.weight'] = imp['f2'].reshape(-1, 1)
        imp_brd['mlp.fc2.fc.bias'] = imp['f2']

        return imp_brd


class AverageActivation(LDAImportance):
    def post_learning(self, expr):
        self.eval()
        imp = {'c1': [],
               'c2': [],
               'c3': [],
               'f1': [],
               'f2': []}
        hs = {'c1': [],
              'c2': [],
              'c3': [],
              'f1': [],
              'f2': []}
        loader = DataLoader(expr.dataset, batch_size=100, shuffle=False)
        for x, t in loader:
            if torch.cuda.is_available():
                x = x.to(torch.device('cuda'))
            output = self(x)
            for key in hs:
                hs[key].append(output['hs'][key].detach())

        for key in hs:
            hs[key] = torch.cat(hs[key], dim=0)
            # deal with CONV output dimensions
            if 'c' in key:
                hs[key] = hs[key].mean(dim=(2, 3))
            imp[key] = hs[key].mean(dim=0)

        self.imp_log.append({n: p.to('cpu') for n, p in imp.items()})
        self.logger.log(importance=self.imp_log)

        imp_brd = self.broadcast_imp(imp)

        anchors = self.get_anchors()
        if self.all_anchors:
            self.reg_params.append((anchors, imp_brd))
        else:
            if len(self.reg_params) > 0:
                prev_imp_brd = self.reg_params[0][1]
                for k in imp_brd:
                    # imp_brd[k] = torch.max(imp_brd[k], prev_imp_brd[k])
                    imp_brd[k] += prev_imp_brd[k]
                    imp_brd[k] *= self.alpha
            self.reg_params = [(anchors, imp_brd)]


class AverageLDA(LDAImportance):
    def __init__(self,
                 encoder,
                 n_penultimate,
                 reg_coef,
                 reg_coef_avg,
                 reg_coef_deeply,
                 alpha,
                 all_anchors=False,
                 cnb=128,
                 fc1nb=1024):
        super(LDAImportance, self).__init__(encoder, n_penultimate)
        self.reg_coef = reg_coef
        self.reg_coef_avg = reg_coef_avg
        self.reg_coef_deeply = reg_coef_deeply
        self.reg_params = []
        self.all_anchors = all_anchors
        self.alpha = alpha
        self.imp_log = []
        self.temp_heads = None
        self.cnb = cnb
        self.fc1nb = fc1nb

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        output = self(x, unit_names)
        loss = self.compute_loss(output['y'], t, unit_names, loss_fn,
                                 pos_weight)
        loss = sum(loss)
        loss += self.reg_coef_deeply * self.deeply_loss(
            output['hs'], t, unit_names
        )
        loss += self.reg_loss()
        loss.backward()

        self.optimizer.step()
        self.logger.increment_batches_seen()
        return loss.item()

    def pre_learning(self, expr):
        device = self.get_current_device()
        self.temp_heads = ModuleDict()
        for key in ['f1']:
            in_units = self.fc1nb
            self.temp_heads.update({
                key: torch.nn.Linear(
                    in_units,
                    len(expr.unit_names)
                ).to(device)
            })

    def deeply_loss(self, hs, t, unit_names):
        dloss = 0.
        for key in hs:
            if key != 'f1':
                continue
            h = hs[key]
            yh = self.temp_heads[key](h)
            tm = map_targets(t, unit_names).to(yh.device)
            dloss += cross_entropy(yh, tm)
        return dloss

    def reg_loss(self):
        loss = []
        for anchors, imps in self.reg_params:
            for n, p in self.encoder.named_parameters():
                if 'conv' in n:
                    reg_coef = self.reg_coef_avg
                else:
                    reg_coef = self.reg_coef

                loss.append(reg_coef * (imps[n] / (imps[n].max() + 1e-9)
                                        * ((p - anchors[n])**2)).sum())
        return sum(loss)

    def post_learning(self, expr):
        self.eval()
        average_imp = {'c1': [],
                       'c2': [],
                       'c3': []}
        hs = {'c1': [],
              'c2': [],
              'c3': [],
              'f1': [],
              'f2': []}
        loader = DataLoader(expr.dataset, batch_size=100, shuffle=False)
        for x, t in loader:
            if torch.cuda.is_available():
                x = x.to(torch.device('cuda'))
            output = self(x)
            for key in hs:
                hs[key].append(output['hs'][key].detach())

        for key in hs:
            hs[key] = torch.cat(hs[key], dim=0)
            # deal with CONV output dimensions
            if 'c' in key:
                hs[key] = hs[key].mean(dim=(2, 3))
                average_imp[key] = hs[key].mean(dim=0)

        tagged_subsets = self.split_tagged_dataset(expr.dataset,
                                                   expr.unit_names)
        # handle more than 2 classes
        assert len(tagged_subsets) == 2
        class_means = {'f1': [],
                       'f2': []}
        class_vars = {'f1': [],
                      'f2': []}
        imp = {'f1': [],
               'f2': []}
        for subset in tagged_subsets:
            hs = {'c1': [],
                  'c2': [],
                  'c3': [],
                  'f1': [],
                  'f2': []}
            loader = DataLoader(subset, batch_size=100, shuffle=False)
            for x, t in loader:
                if torch.cuda.is_available():
                    x = x.to(torch.device('cuda'))
                output = self(x)
                for key in hs:
                    hs[key].append(output['hs'][key].detach())

            for key in hs:
                if 'f' in key:
                    hs[key] = torch.cat(hs[key], dim=0)
                    class_means[key].append(hs[key].mean(dim=0))
                    class_vars[key].append(hs[key].var(dim=0))
                # deal with CONV output dimensions
                # hs[key] = hs[key].mean(dim=(2, 3))

        for key in imp:
            imp[key] = self.compute_2class_imp(
                class_means[key], class_vars[key])

        imp.update(average_imp)

        self.imp_log.append({n: p.to('cpu') for n, p in imp.items()})
        self.logger.log(importance=self.imp_log)

        imp_brd = self.broadcast_imp(imp)

        anchors = self.get_anchors()
        if self.all_anchors:
            self.reg_params.append((anchors, imp_brd))
        else:
            if len(self.reg_params) > 0:
                prev_imp_brd = self.reg_params[0][1]
                for k in imp_brd:
                    # imp_brd[k] = torch.max(imp_brd[k], prev_imp_brd[k])
                    imp_brd[k] += prev_imp_brd[k]
                    imp_brd[k] *= self.alpha
            self.reg_params = [(anchors, imp_brd)]


class EWC(L2):
    def __init__(self,
                 encoder,
                 n_penultimate,
                 reg_coef,
                 all_anchors,
                 fisher_samples=1000,
                 empirical_fisher=True,
                 simple_fisher=False,
                 fisher_loss_fn=None,
                 clip_gradients=True,
                 include_unknown=True):
        super(EWC, self).__init__(encoder,
                                  n_penultimate,
                                  reg_coef,
                                  all_anchors, clip_gradients)
        self.fisher_samples = fisher_samples
        self.empirical_fisher = empirical_fisher
        self.simple_fisher = simple_fisher
        self.fisher_loss_fn = fisher_loss_fn
        self.imp_log = []
        self.include_unknown = include_unknown

    def post_learning(self, expr):
        anchors = self.get_anchors()
        if self.simple_fisher:
            fishers = self.get_fishers_simple(expr)
        else:
            fishers = self.get_fishers(expr)
        if self.all_anchors:
            self.reg_params.append((anchors, fishers))
        else:
            for _, f in self.reg_params:  # len(self.reg_params) =< 1
                for n in fishers:
                    fishers[n] += f[n]
            self.reg_params = [(anchors, fishers)]

        # fishers = self.normalize(fishers)
        self.imp_log.append((
            fishers['conv.conv1.conv.weight'].view(
                fishers['conv.conv1.conv.weight'].shape[0], -1
            ).norm(dim=1).to('cpu'),
            fishers['conv.conv2.conv.weight'].view(
                fishers['conv.conv2.conv.weight'].shape[0], -1
            ).norm(dim=1).to('cpu'),
            fishers['conv.conv3.conv.weight'].view(
                fishers['conv.conv3.conv.weight'].shape[0], -1
            ).norm(dim=1).to('cpu'),
            fishers['mlp.fc1.fc.weight'].norm(dim=1).to('cpu'),
            fishers['mlp.fc2.fc.weight'].norm(dim=1).to('cpu'),
        ))
        self.logger.log(importance=self.imp_log)

    def get_fishers(self, expr):
        fishers = {
            n: torch.zeros_like(p)
            for n, p in self.encoder.named_parameters()
        }
        loader = DataLoader(expr.dataset, batch_size=1, shuffle=False)
        self.activate(expr.task_name)
        self.eval()
        count = 0
        if self.fisher_samples is None:
            fisher_samples = len(loader)
        else:
            fisher_samples = self.fisher_samples
        done = False
        device = next(self.parameters()).device

        if self.fisher_loss_fn is None:
            fisher_loss_fn = expr.loss_fn
        elif self.fisher_loss_fn == 'multi':
            fisher_loss_fn = 'multi'
        else:
            fisher_loss_fn = 'binary'

        while True:
            for x, t in loader:
                x = x.to(device)
                y = self(x, unit_names=expr.unit_names)['y']
                log_ps = self.log_p(y, t[0], fisher_loss_fn, expr.unit_names)
                for log_p in log_ps:
                    self.zero_grad()
                    log_p.backward(retain_graph=True)
                    for n, p in self.encoder.named_parameters():
                        fishers[n] += deepcopy(p.grad.detach())**2
                count += 1
                if count == fisher_samples:
                    done = True
                    break
            if done:
                break
        for n, p in self.encoder.named_parameters():
            fishers[n] /= count
        return fishers

    def get_fishers_simple(self, expr):
        fishers = {
            n: torch.zeros_like(p)
            for n, p in self.encoder.named_parameters()
        }
        loader = DataLoader(expr.dataset, batch_size=1, shuffle=False)
        self.activate(expr.task_name)
        self.eval()
        count = 0
        if self.fisher_samples is None:
            fisher_samples = len(loader)
        else:
            fisher_samples = self.fisher_samples
        done = False
        device = next(self.parameters()).device
        while True:
            for x, t in loader:
                if t[0] == 'unknown' and not self.include_unknown:
                    continue
                self.zero_grad()
                x = x.to(device)
                y = self(x, unit_names=expr.unit_names)['y']
                loss = self.compute_loss(y,
                                         t,
                                         expr.unit_names,
                                         loss_fn='multi',
                                         pos_weight=1.0)
                loss = sum(loss)
                loss.backward()

                for n, p in self.encoder.named_parameters():
                    fishers[n] += deepcopy(p.grad.detach())**2

                count += 1
                if count == fisher_samples:
                    done = True
                    break
            if done:
                break
        for n, p in self.encoder.named_parameters():
            fishers[n] /= count
        return fishers

    def log_p(self, y, t, loss_fn, unit_names):
        if loss_fn == 'binary':
            log_ps = []
            for u in unit_names:
                if not self.empirical_fisher:
                    t_u = int(torch.bernoulli(torch.sigmoid(y[u])).item())
                else:
                    t_u = int(t == u)
                if t_u == 0:
                    y_logsg = logsigmoid(y[u]) - y[u]  # log(1 - sigmoid(y[u]))
                else:
                    y_logsg = logsigmoid(y[u])
                log_ps.append(y_logsg)
            return log_ps

        else:
            y = torch.cat([y[u] for u in unit_names], dim=1)
            y_logsm = log_softmax(y, dim=1)
            if not self.empirical_fisher:
                y_sm = softmax(y, dim=1)
                t = torch.multinomial(y_sm, 1).item()
            else:
                t = unit_names.index(t)
            return [y_logsm[0, t]]

    def reg_loss(self):
        loss = []
        for anchors, fishers in self.reg_params:
            # fishers = self.normalize(fishers)
            for n, p in self.encoder.named_parameters():
                loss.append((fishers[n] * ((p - anchors[n])**2)).sum())
        return sum(loss)

    @staticmethod
    def normalize(tensor_dict):
        max_value = 1e-12
        for p in tensor_dict.values():
            p_max = p.max().item()
            if p_max > max_value:
                max_value = p_max
        for n in tensor_dict:
            tensor_dict[n] /= p_max
        return tensor_dict


class LwF(Vanilla):
    def __init__(self, encoder, n_penultimate, reg_coef, batch_size):
        super(LwF, self).__init__(encoder, n_penultimate)
        self.reg_coef = reg_coef
        self.batch_size = batch_size
        self.prev_outputs = None
        self.batch_counter = 0
        self.nb_batches = 0

    def pre_learning(self, expr):
        if len(self.output_layers) == 0:
            return

        loader = DataLoader(expr.dataset,
                            batch_size=self.batch_size,
                            shuffle=False)

        self.reset_training_loader(loader)

        device = self.get_current_device()
        self.eval()
        ts = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                t = {}
                z = self.encoder(x)
                for task_name in self.output_layers:
                    t[task_name] = self.output_layers[task_name](z)
                ts.append(t)
            self.prev_outputs = ts

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        z = self.encoder(x)
        y = self.output_layers[self.active_layer](z)
        loss = self.compute_loss(y, t, unit_names, loss_fn, pos_weight)
        loss = sum(loss)

        if self.prev_outputs is not None:
            loss += self.reg_coef * self.reg_loss(z)
            self.batch_counter += 1
            if self.batch_counter == self.nb_batches:
                self.batch_counter = 0

        loss.backward()
        self.optimizer.step()
        self.logger.increment_batches_seen()
        return loss.item()

    def reg_loss(self, penultimate_input):
        if self.prev_outputs is None:
            raise Exception('No previous outputs!')

        y_olds = self.prev_outputs[self.batch_counter]
        rloss = 0.
        for task_name in self.output_layers:
            if task_name not in y_olds:
                continue
            y_old = y_olds[task_name]
            us = tuple([u for u in y_old])  # names of units to regularize
            y_new = self.output_layers[task_name](penultimate_input, us)

            y_old = torch.cat([y_old[u] for u in us], dim=1)
            y_new = torch.cat([y_new[u] for u in us], dim=1)

            rloss += distillation_categorical(y_new, y_old)
        return rloss

    def reset_training_loader(self, loader):
        self.batch_counter = 0
        self.nb_batches = len(loader)


class EWCpp(L2):
    def __init__(self,
                 encoder,
                 n_penultimate,
                 reg_coef,
                 fisher_alpha,
                 clip_gradients=False):
        super(EWCpp, self).__init__(encoder,
                                    n_penultimate,
                                    reg_coef,
                                    all_anchors=False,
                                    clip_gradients=clip_gradients)
        self.fisher_alpha = fisher_alpha
        self.running_fisher = None

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        output = self(x, unit_names)
        loss = self.compute_loss(output['y'], t, unit_names, loss_fn,
                                 pos_weight)
        loss = sum(loss)
        loss.backward()
        self.update_running_fisher()

        if len(self.reg_params) > 0:
            rloss = self.reg_coef * self.reg_loss()
            rloss.backward()
        else:
            rloss = 0.

        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.parameters(),
                                           0.1)

        self.optimizer.step()
        self.logger.increment_batches_seen()
        return (loss + rloss).item()

    def get_batch_fisher(self):
        return {
            n: deepcopy(p.grad.detach()) ** 2
            for n, p in self.encoder.named_parameters()
        }

    def update_running_fisher(self):
        batch_fisher = self.get_batch_fisher()
        if self.running_fisher is None:
            self.running_fisher = batch_fisher
        else:
            for n in self.running_fisher:
                self.running_fisher[n] = (
                    self.fisher_alpha * batch_fisher[n] +
                    (1 - self.fisher_alpha) * self.running_fisher[n]
                )

    def post_learning(self, expr):
        anchors = self.get_anchors()
        fishers = deepcopy(self.running_fisher)
        self.reg_params = [(anchors, fishers)]

    def reg_loss(self):
        loss = []
        for anchors, fishers in self.reg_params:
            for n, p in self.encoder.named_parameters():
                loss.append((fishers[n] * ((p - anchors[n])**2)).sum())
        return sum(loss)


class RWalk(EWCpp):
    def __init__(
        self,
        encoder,
        n_penultimate,
        reg_coef,
        fisher_alpha,
        clip_gradients=False
    ):
        super(RWalk, self).__init__(
            encoder,
            n_penultimate,
            reg_coef,
            fisher_alpha,
            clip_gradients=False
        )
        self.running_importance = None
        self.prev_importance = None
        self.prev_step_loss_grad = None
        self.prev_step_anchors = None
        self.prev_step_fisher = None

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        output = self(x, unit_names)
        loss = self.compute_loss(output['y'], t, unit_names, loss_fn,
                                 pos_weight)
        loss = sum(loss)
        loss.backward()
        self.update_running_fisher()

        if len(self.reg_params) > 0:
            rloss = self.reg_coef * self.reg_loss()
            rloss.backward()
        else:
            rloss = 0.

        self.update_importance()

        if self.clip_gradients:
            torch.nn.utils.clip_grad_norm_(self.parameters(),
                                           0.1)

        self.optimizer.step()
        self.logger.increment_batches_seen()
        return (loss + rloss).item()

    def post_learning(self, expr):
        anchors = self.get_anchors()
        fishers = deepcopy(self.running_fisher)
        importance = deepcopy(self.running_importance)
        if self.prev_importance is None:
            self.prev_importance = {}
            for n, p in importance.items():
                p[p < 0] = 0.
                self.prev_importance[n] = p
                self.running_importance[n] *= 0.
        else:
            for n, p in self.prev_importance.items():
                importance[n][importance[n] < 0] = 0.
                self.prev_importance[n] = (
                    p + importance[n]
                ) / 2
                self.running_importance[n] *= 0.
        importance = self.normalize(deepcopy(self.prev_importance))
        fishers = self.normalize(fishers)
        rwalk_coefs = {}
        for n, p in importance.items():
            rwalk_coefs[n] = p + fishers[n]
        self.reg_params = [(anchors, rwalk_coefs)]

    def update_importance(self):
        if self.prev_step_loss_grad is not None:
            importance = self.step_importance()
            if self.running_importance is None:
                self.running_importance = importance
            else:
                for n in self.running_importance:
                    self.running_importance[n] += importance[n]

        self.prev_step_loss_grad = {
            n: deepcopy(p.grad.detach())
            for n, p in self.encoder.named_parameters()
        }

        self.prev_step_anchors = {
            n: deepcopy(p.detach()) for n, p in self.encoder.named_parameters()
        }

    def step_importance(self):
        importance = {}
        for n, p in self.encoder.named_parameters():
            num = (
                -self.prev_step_loss_grad[n] *
                (deepcopy(p.detach()) - self.prev_step_anchors[n])
            )
            den = (
                0.5 * self.running_fisher[n] *
                ((deepcopy(p.detach()) - self.prev_step_anchors[n]) ** 2)
            )
            importance[n] = num / (den + 1e-9)
        return importance

    @staticmethod
    def normalize(tensor_dict):
        max_value = 1e-10
        for p in tensor_dict.values():
            p_max = p.max().item()
            if p_max > max_value:
                max_value = p_max
        for n in tensor_dict:
            tensor_dict[n] /= p_max
        return tensor_dict


class Replay(ContinualLearningModel):
    def __init__(self, encoder, n_penultimate,
                 memory_budget, batch_size, replay_period=1,
                 replay_epochs=1):
        super(Replay, self).__init__(encoder, n_penultimate)
        self.memory = {}
        self.memory_budget = memory_budget
        self.replay_period = replay_period
        self.batch_size = batch_size
        self.replay_epochs = replay_epochs

    def post_learning(self, expr):
        self.update_memory(expr)

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        y = self(x, unit_names=unit_names)['y']
        loss = self.compute_loss(y, t, unit_names, loss_fn, pos_weight)
        loss = sum(loss)
        loss.backward()
        self.optimizer.step()
        self.logger.increment_batches_seen()

        if self.logger.batches_seen % self.replay_period == 0:
            self.replay(loss_fn)

        return loss.item()

    def update_memory(self, expr):
        candidate_dataset = TaggedDataset.from_tagged_dataset(
            expr.dataset, self.memory_budget
        )
        unit_names = expr.unit_names
        if expr.task_name not in self.memory:
            self.memory[expr.task_name] = (unit_names, candidate_dataset)
        else:
            candidate_dataset.extend(self.memory[expr.task_name][1])
            unit_names += self.memory[expr.task_name][0]
            unit_names = tuple(set(unit_names))

            self.memory[expr.task_name] = (
                unit_names,
                candidate_dataset
            )

    def replay(self, loss_fn, x_batch=None, scalec=1.0):
        if not self.memory:
            print('Memory empty. Skipping replay.')
            return
        else:
            # replay
            self.train()
            prev_active_layer = self.active_layer
            for task_name in self.memory:
                self.activate(task_name)
                replay_loader = DataLoader(self.memory[task_name][1],
                                           batch_size=self.batch_size,
                                           shuffle=True)

                for _ in range(self.replay_epochs):
                    for x, t in replay_loader:
                        self.optimizer.zero_grad()
                        if torch.cuda.is_available():
                            x = x.to(torch.device('cuda'))
                        # unit_names = tuple(
                        #     [name for name in
                        #      self.output_layers[self.active_layer].output_units]
                        # )
                        if x_batch is not None:
                            scale = self.avg_pairwise(x_batch, x)
                            print(scale)
                            scale = (scale - 0.35) / (0.1)
                            scale = torch.max(torch.tensor(0.), scale)
                            print(scale, '\n')
                        else:
                            scale = 1.0
                        unit_names = self.memory[task_name][0]
                        y = self(x, unit_names)['y']
                        loss = self.compute_loss(
                            y, t, unit_names, loss_fn=loss_fn, pos_weight=1.0)
                        loss = scalec * sum(loss) * scale
                        loss.backward()
                        self.optimizer.step()
            self.activate(prev_active_layer)

    def dist_to_batch(self, t, batch):
        return [torch.dist(t, other).item() for other in batch]

    def avg_min_dist(self, batch, other_batch):
        dist = [min(self.dist_to_batch(t, other_batch)) for t in batch]
        return sum(dist) / len(dist)

    def sum_square_dist(self, batch, other_batch):
        scale = batch.shape[0] * other_batch.numel()
        return sum([torch.dist(t, other_batch).pow(2) for t in batch]) / scale

    def pairwise(self, t, batch):
        nb = batch.shape[0]
        return torch.nn.functional.pairwise_distance(
            t.reshape(1, -1).repeat(nb, 1),
            batch.reshape(nb, -1)
        ).min() / np.sqrt(t.numel())

    def avg_pairwise(self, batch, other_batch):
        return torch.tensor([self.pairwise(t, other_batch)
                             for t in batch]).mean()


class ReplayLwF(Replay):
    def __init__(self, encoder, n_penultimate, reg_coef,
                 batch_size, memory_budget,
                 replay_period=1, replay_epochs=1, scale_replay=False,
                 scalec=1.0):
        super(ReplayLwF, self).__init__(encoder,
                                        n_penultimate,
                                        memory_budget,
                                        batch_size,
                                        replay_period,
                                        replay_epochs)
        self.reg_coef = reg_coef
        self.prev_outputs = None
        self.batch_counter = 0
        self.nb_batches = 0
        self.scale_replay = scale_replay
        self.scalec = scalec

    def pre_learning(self, expr):
        if len(self.output_layers) == 0:
            return

        loader = DataLoader(expr.dataset,
                            batch_size=self.batch_size,
                            shuffle=False)

        self.reset_training_loader(loader)

        device = self.get_current_device()
        self.eval()
        ts = []
        with torch.no_grad():
            for x, _ in loader:
                x = x.to(device)
                t = {}
                z = self.encoder(x)
                for task_name in self.output_layers:
                    t[task_name] = self.output_layers[task_name](z)
                ts.append(t)
            self.prev_outputs = ts

    def reg_loss(self, penultimate_input):
        if self.prev_outputs is None:
            raise Exception('No previous outputs!')

        y_olds = self.prev_outputs[self.batch_counter]
        rloss = 0.
        for task_name in self.output_layers:
            if task_name not in y_olds:
                continue
            y_old = y_olds[task_name]
            us = tuple([u for u in y_old])  # names of units to regularize
            y_new = self.output_layers[task_name](penultimate_input, us)

            y_old = torch.cat([y_old[u] for u in us], dim=1)
            y_new = torch.cat([y_new[u] for u in us], dim=1)

            rloss += distillation_categorical(y_new, y_old)
        return rloss

    def reset_training_loader(self, loader):
        self.batch_counter = 0
        self.nb_batches = len(loader)

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        z = self.encoder(x)
        y = self.output_layers[self.active_layer](z)
        loss = self.compute_loss(y, t, unit_names, loss_fn, pos_weight)
        loss = sum(loss)

        if self.prev_outputs is not None:
            loss += self.reg_coef * self.reg_loss(z)
            self.batch_counter += 1
            if self.batch_counter == self.nb_batches:
                self.batch_counter = 0

        loss.backward()
        self.optimizer.step()
        self.logger.increment_batches_seen()

        if self.logger.batches_seen % self.replay_period == 0:
            x_batch = x if self.scale_replay else None
            self.replay(loss_fn, x_batch=x_batch, scalec=self.scalec)
        return loss.item()


class ReplayEWC(Replay):
    def __init__(self, encoder, n_penultimate, reg_coef,
                 batch_size, memory_budget,
                 replay_period=1, replay_epochs=1, scale_replay=False,
                 scalec=1.0,
                 all_anchors=True,
                 fisher_samples=1000):
        super(ReplayEWC, self).__init__(encoder,
                                        n_penultimate,
                                        memory_budget,
                                        batch_size,
                                        replay_period,
                                        replay_epochs)
        self.reg_coef = reg_coef
        self.prev_outputs = None
        self.batch_counter = 0
        self.nb_batches = 0
        self.scale_replay = scale_replay
        self.scalec = scalec
        self.fisher_samples = fisher_samples
        self.all_anchors = all_anchors
        self.reg_params = []

    def get_anchors(self):
        return {
            n: deepcopy(p.detach())
            for n, p in self.encoder.named_parameters()
        }

    def post_learning(self, expr):
        self.update_memory(expr)

        anchors = self.get_anchors()
        fishers = self.get_fishers_simple(expr)

        if self.all_anchors:
            self.reg_params.append((anchors, fishers))
        else:
            for _, f in self.reg_params:  # len(self.reg_params) =< 1
                for n in fishers:
                    fishers[n] += f[n]
            self.reg_params = [(anchors, fishers)]

    def get_fishers_simple(self, expr):
        fishers = {
            n: torch.zeros_like(p)
            for n, p in self.encoder.named_parameters()
        }
        loader = DataLoader(expr.dataset, batch_size=1, shuffle=False)
        self.activate(expr.task_name)
        self.eval()
        count = 0
        if self.fisher_samples is None:
            fisher_samples = len(loader)
        else:
            fisher_samples = self.fisher_samples
        done = False
        device = next(self.parameters()).device
        while True:
            for x, t in loader:
                self.zero_grad()
                x = x.to(device)
                y = self(x, unit_names=expr.unit_names)['y']
                loss = self.compute_loss(y,
                                         t,
                                         expr.unit_names,
                                         loss_fn='multi',
                                         pos_weight=1.0)
                loss = sum(loss)
                loss.backward()

                for n, p in self.encoder.named_parameters():
                    fishers[n] += deepcopy(p.grad.detach())**2

                count += 1
                if count == fisher_samples:
                    done = True
                    break
            if done:
                break
        for n, p in self.encoder.named_parameters():
            fishers[n] /= count
        return fishers

    def reg_loss(self):
        loss = []
        for anchors, fishers in self.reg_params:
            for n, p in self.encoder.named_parameters():
                loss.append((fishers[n] * ((p - anchors[n])**2)).sum())
        return sum(loss)

    def fit_batch(self, x, t, unit_names, loss_fn, pos_weight=1.0):
        self.train()
        self.optimizer.zero_grad()
        z = self.encoder(x)
        y = self.output_layers[self.active_layer](z)
        loss = self.compute_loss(y, t, unit_names, loss_fn, pos_weight)
        loss = sum(loss)

        loss += self.reg_coef * self.reg_loss()

        loss.backward()
        self.optimizer.step()
        self.logger.increment_batches_seen()

        if self.logger.batches_seen % self.replay_period == 0:
            x_batch = x if self.scale_replay else None
            self.replay(loss_fn, x_batch=x_batch, scalec=self.scalec)
        return loss.item()
