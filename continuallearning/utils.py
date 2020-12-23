import torch
from torch.nn.functional import cross_entropy, logsigmoid, softmax
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.nn.functional import log_softmax
from torch.utils.data import TensorDataset
from random import shuffle
from pathlib import Path


def split_tensor_dataset(dataset, ratio):
    cutoff = int(ratio * len(dataset))
    data, labels = dataset.tensors
    index = list(range(len(dataset)))
    shuffle(index)
    index = torch.tensor(index)
    dataset_1 = TensorDataset(
        torch.index_select(data, 0, index[:cutoff]),
        torch.index_select(labels, 0, index[:cutoff]))
    dataset_2 = TensorDataset(
        torch.index_select(data, 0, index[cutoff:]),
        torch.index_select(labels, 0, index[cutoff:]))
    return dataset_1, dataset_2


def get_subset(dataset, classes, samples_per_class=None):
    data, labels = dataset.tensors
    indices = [i for i, t in enumerate(labels) if t.item() in classes]
    shuffle(indices)
    if samples_per_class is not None:
        indices = indices[:samples_per_class * len(classes)]
    indices = torch.tensor(indices)
    data = torch.index_select(data, 0, indices)
    labels = torch.index_select(labels, 0, indices)
    return TensorDataset(data, labels)


def torchvision_to_tensor(dataset):
    assert isinstance(dataset[0][0], torch.Tensor)
    assert isinstance(dataset[0][1], torch.Tensor)
    assert len(dataset[0][1].shape) == 0
    data = torch.cat([dataset[i][0].unsqueeze(0) for i in range(len(dataset))])
    labels = torch.tensor([dataset[i][1].item() for i in range(len(dataset))],
                          dtype=torch.long)
    return TensorDataset(data, labels)


def map_targets(tagged_targets, unit_names):
    if not isinstance(unit_names, tuple):
        raise Exception('expected unit_names to be tuple')

    if not isinstance(tagged_targets, tuple):
        raise Exception('expected tagged_targets to be tuple')

    t = list(map(lambda i: unit_names.index(i), tagged_targets))
    t = torch.tensor(t).long()
    return t


def dict_cross_entropy(y, t, unit_names):
    t = map_targets(t, unit_names)
    y = torch.cat([y[u] for u in unit_names], dim=1)
    t = t.to(y.device)
    return [cross_entropy(y, t)]


def dict_binary_entropy(y,
                        t,
                        unit_names,
                        pos_weight=1.0):
    loss = []
    device = list(y.values())[0].device
    pos_weight = torch.tensor([pos_weight]).float().to(device)
    for u in unit_names:
        if u == 'unknown':
            continue
        t_u = torch.tensor([ti == u for ti in t])
        t_u = t_u.float().view(-1, 1)
        t_u = t_u.to(device)
        loss.append(
            binary_cross_entropy_with_logits(y[u], t_u, pos_weight=pos_weight)
        )
    return loss


def distillation_binary(y, t):
    t = torch.sigmoid(t / 2)
    t = torch.stack([t, 1 - t], dim=1)

    y /= 2
    log_y = logsigmoid(y)
    log_y = torch.stack([log_y, log_y - y], dim=1)

    return -1 * (t, log_y).sum() / t.shape[0]


def distillation_categorical(y, t):
    t = softmax(t / 2, dim=1)
    log_y = log_softmax(y / 2, dim=1)
    return -1 * (t * log_y).sum() / t.shape[0]


def expand(tensor):
    return tensor.expand(3, -1, -1)


def load_segment_as_tensor_dataset(segment, dataset_name, root='data'):
    root = Path(root)
    filename = f'{dataset_name}-{segment}.pt'
    dataset_path = root / 'tensor_datasets' / filename
    return torch.load(dataset_path)


def kl_divergence(e1, e2, X, h):
    Y1 = h(e1(X))
    Y2 = h(e2(X))
    Y1_sm = torch.nn.functional.softmax(Y1, dim=1)
    Y1_logsm = torch.nn.functional.log_softmax(Y1, dim=1)
    Y2_logsm = torch.nn.functional.log_softmax(Y2, dim=1)
    ts = torch.multinomial(Y1_sm, 10000, replacement=True)
    yg1 = torch.gather(Y1_logsm, 1, ts)
    yg2 = torch.gather(Y2_logsm, 1, ts)
    d = yg1 - yg2
    return (d.sum() / d.numel()).to('cpu').item()


def kl_divergences(es, X, n_penultimate):
    h = torch.nn.Linear(n_penultimate, 10)
    if torch.cuda.is_available():
        h.to('cuda')
        X = X.to('cuda')
    kl1 = []
    kl2 = []

    for i in range(len(es) - 1):
        kl1.append(kl_divergence(es[i], es[i+1], X, h))

    for i in range(len(es) - 2):
        kl2.append(kl_divergence(es[i], es[i+2], X, h))

    klf = kl_divergence(es[2], es[-1], X, h)

    return kl1, kl2, klf
