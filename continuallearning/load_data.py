import json
from pathlib import Path
from random import shuffle
from collections import OrderedDict
import torch
from torch.utils.data import Dataset
from continuallearning.prepare_data import prepare_dataset
from continuallearning.utils import get_subset, load_segment_as_tensor_dataset


class TaggedDataset(Dataset):
    def __init__(self, inputs, tagged_targets):
        assert inputs.size(0) == len(tagged_targets)
        self.inputs = inputs
        self.tagged_targets = tagged_targets

    def __getitem__(self, index):
        return self.inputs[index], self.tagged_targets[index]

    def __len__(self):
        return self.inputs.size(0)

    def extend(self, other):
        self.inputs = torch.cat((self.inputs, other.inputs), dim=0)
        self.tagged_targets = self.tagged_targets + other.tagged_targets
        assert self.inputs.size(0) == len(self.tagged_targets)

    def shuffle(self):
        index = list(range(len(self.tagged_targets)))
        shuffle(index)
        self.tagged_targets = tuple([self.tagged_targets[i] for i in index])
        self.inputs = torch.index_select(self.inputs, 0, torch.tensor(index))

    @classmethod
    def from_tagged_dataset(cls, tagged_dataset, sample_size):
        index = list(range(len(tagged_dataset)))
        shuffle(index)
        sample_tagged_targets = tuple(
            [tagged_dataset.tagged_targets[i] for i in index]
        )
        index = torch.tensor(index)
        sample_inputs = torch.index_select(tagged_dataset.inputs, 0, index)
        return cls(sample_inputs[:sample_size],
                   sample_tagged_targets[:sample_size])

    @classmethod
    def from_tensor_dataset(cls,
                            dataset,
                            tag,
                            classes=None,
                            samples_per_class=None):

        if classes is not None:
            dataset = get_subset(dataset, classes, samples_per_class)

        inputs, targets = dataset.tensors
        if tag == 'unknown':
            tagged_targets = [tag for _ in targets.view(-1)]
        else:
            tagged_targets = [
                '{}-{:d}'.format(tag, int(t.item())) for t in targets.view(-1)
            ]
        tagged_targets = tuple(tagged_targets)
        return cls(inputs, tagged_targets)


class LearningExperience(object):
    def __init__(self,
                 task_name,
                 expr_id,
                 dataset: TaggedDataset,
                 loss_fn: str,
                 pos_weight=1.0):

        self.task_name = task_name
        self.id = expr_id
        self.dataset = dataset
        self.loss_fn = loss_fn
        self.pos_weight = pos_weight

        unit_names = []
        for x, t in dataset:
            if t not in unit_names:
                unit_names.append(t)
        self.unit_names = tuple(unit_names)

    @classmethod
    def from_config(cls,
                    expr_id,
                    expr_config,
                    loss_fn,
                    pos_weight=1.0,
                    samples_per_class=None,
                    aux_samples_per_class=None,
                    root='data'):
        task_name, subsets = expr_config

        tagged_dataset = None
        for dataset_name, classes in subsets['train']:
            prepare_dataset(dataset_name, root)
            dataset = load_segment_as_tensor_dataset('train',
                                                     dataset_name,
                                                     root)
            td = TaggedDataset.from_tensor_dataset(
                dataset,
                tag=dataset_name,
                classes=classes,
                samples_per_class=samples_per_class
            )
            if tagged_dataset is None:
                tagged_dataset = td
            else:
                tagged_dataset.extend(td)

        # augment with auxiliary data if available
        if 'aux' in subsets:
            for dataset_name, classes, segment in subsets['aux']:
                prepare_dataset(dataset_name, root)
                dataset = load_segment_as_tensor_dataset(segment,
                                                         dataset_name,
                                                         root)
                tagged_dataset.extend(
                    TaggedDataset.from_tensor_dataset(
                        dataset,
                        tag='unknown',
                        classes=classes,
                        samples_per_class=aux_samples_per_class
                    )
                )

        tagged_dataset.shuffle()

        return cls(task_name=task_name,
                   expr_id=expr_id,
                   dataset=tagged_dataset,
                   loss_fn=loss_fn,
                   pos_weight=pos_weight)


class EvaluationUnit(object):
    def __init__(self, tasks):
        self.evu_tasks = tasks

    @classmethod
    def from_config(cls,
                    evu_config,
                    dev=True,
                    samples_per_class=None,
                    root='data'):
        segment = 'dev' if dev else 'test'
        tasks = []
        for task_name, subset_defs in evu_config.items():
            tagged_dataset = None
            reporting_order = OrderedDict()
            for expr_id, expr_subset_defs in subset_defs.items():
                expr_unit_names = []
                for dataset_name, classes in expr_subset_defs:
                    expr_unit_names.extend(
                        [f'{dataset_name}-{c:d}' for c in classes]
                    )
                    prepare_dataset(dataset_name, root)
                    dataset = load_segment_as_tensor_dataset(segment,
                                                             dataset_name,
                                                             root)
                    td = TaggedDataset.from_tensor_dataset(
                        dataset,
                        tag=dataset_name,
                        classes=classes,
                        samples_per_class=samples_per_class
                    )
                    if tagged_dataset is None:
                        tagged_dataset = td
                    else:
                        tagged_dataset.extend(td)

                reporting_order[expr_id] = tuple(expr_unit_names)
            tagged_dataset.shuffle()
            tasks.append((task_name, tagged_dataset, reporting_order))
        return cls(tasks)


class Episode(object):
    def __init__(self, exprs, evus):
        assert len(exprs) == len(evus)
        self.exprs = exprs
        self.evus = evus

    def __iter__(self):
        yield from zip(self.exprs, self.evus)

    @staticmethod
    def generate_from_json(json_path,
                           loss_fn,
                           dev=True,
                           train_samples_per_class=None,
                           test_samples_per_class=None,
                           aux_samples_per_class=None,
                           pos_weight=1.0,
                           root='data'):
        json_path = Path(json_path)
        with open(json_path, 'r') as f:
            config = json.load(f)

        evu_config = OrderedDict()
        for expr_id, expr_config in enumerate(config):
            expr = LearningExperience.from_config(expr_id,
                                                  expr_config,
                                                  loss_fn,
                                                  pos_weight,
                                                  train_samples_per_class,
                                                  aux_samples_per_class,
                                                  root)
            task_name, subse_defs = expr_config
            if task_name not in evu_config:
                # ignore aux data in evus
                evu_config[task_name] = OrderedDict([
                    (expr_id, subse_defs['train'])
                ])
            else:
                evu_config[task_name][expr_id] = subse_defs['train']
            evu = EvaluationUnit.from_config(evu_config,
                                             dev,
                                             test_samples_per_class,
                                             root)
            yield expr, evu


def create_baseline_episode(episode_config,
                            dev=True,
                            train_samples_per_class=None,
                            test_samples_per_class=None,
                            root='data'):
    """Creates a single expr episode from an episode_config.
    Single expr (combining all exprs) and as many evus as there
    are exprs in original episode_config. Each evu has data from
    the corresponding expr only. This is used to compute intrasigence.
    An additional evu is added for the combined dataset, to compute
    baseline performance.
    NOTE: Assumes only a single task!!
    """
    json_path = Path(episode_config)
    with open(json_path, 'r') as f:
        config = json.load(f)
    combined_subset_defs = []
    evu_config = OrderedDict()
    for expr_id, (task_name, subset_defs) in enumerate(config):
        if task_name not in evu_config:
            evu_config[task_name] = OrderedDict()
        combined_subset_defs.extend(subset_defs['train'])
        evu_config[task_name][expr_id] = subset_defs['train']
    expr_config = task_name, {'train': combined_subset_defs}
    expr = LearningExperience.from_config(
        expr_id='combined',
        expr_config=expr_config,
        loss_fn='multi',
        samples_per_class=train_samples_per_class,
        root=root
    )
    evu = EvaluationUnit.from_config(
        evu_config=evu_config,
        dev=dev,
        samples_per_class=test_samples_per_class,
        root=root
    )
    return Episode([expr], [evu])
