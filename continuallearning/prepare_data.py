from pathlib import Path
import torch
from torchvision.transforms import ToTensor, Compose, Resize, Lambda
from torchvision.datasets import CIFAR100, MNIST, STL10
from torchvision.datasets import FashionMNIST, KMNIST, CIFAR10, SVHN
from continuallearning.utils import torchvision_to_tensor
from continuallearning.utils import split_tensor_dataset, expand


def prepare_kmnist(root='data'):
    root = Path(root)
    tensor_datasets_path = root / 'tensor_datasets'
    if not tensor_datasets_path.is_dir():
        tensor_datasets_path.mkdir(parents=True)

    not_ready = []
    for segment in ['train', 'test', 'dev']:
        if (tensor_datasets_path / f'kmnist-{segment}.pt').exists():
            print(f'KMNIST {segment} tensor dataset ready')
        else:
            not_ready.append(segment)
    if len(not_ready) == 0:
        return

    torchvision_dataset_path = root / 'torchvision_datasets' / 'kmnist'
    if not torchvision_dataset_path.is_dir():
        torchvision_dataset_path.mkdir(parents=True)

    if 'train' in not_ready or 'dev' in not_ready:
        train_dataset = KMNIST(
            torchvision_dataset_path,
            train=True,
            download=True,
            transform=Compose([Resize((32, 32)),
                               ToTensor(),
                               Lambda(expand)]),
            target_transform=lambda t: torch.tensor(t)
        )
        train_dataset = torchvision_to_tensor(train_dataset)
        dev_dataset, train_dataset = split_tensor_dataset(train_dataset, 0.2)
        torch.save(train_dataset, tensor_datasets_path / 'kmnist-train.pt')
        print('KMNIST train tensor dataset ready')

        torch.save(dev_dataset, tensor_datasets_path / 'kmnist-dev.pt')
        print('KMNIST dev tensor dataset ready')

    if 'test' in not_ready:
        test_dataset = KMNIST(
            torchvision_dataset_path,
            train=False,
            download=True,
            transform=Compose([Resize((32, 32)),
                               ToTensor(),
                               Lambda(expand)]),
            target_transform=lambda t: torch.tensor(t)
        )
        test_dataset = torchvision_to_tensor(test_dataset)
        torch.save(test_dataset, tensor_datasets_path / 'kmnist-test.pt')
        print('KMNIST test tensor dataset ready')


def prepare_fmnist(root='data'):
    root = Path(root)
    tensor_datasets_path = root / 'tensor_datasets'
    if not tensor_datasets_path.is_dir():
        tensor_datasets_path.mkdir(parents=True)

    not_ready = []
    for segment in ['train', 'test', 'dev']:
        if (tensor_datasets_path / f'fmnist-{segment}.pt').exists():
            print(f'FashionMNIST {segment} tensor dataset ready')
        else:
            not_ready.append(segment)
    if len(not_ready) == 0:
        return

    torchvision_dataset_path = root / 'torchvision_datasets' / 'fmnist'
    if not torchvision_dataset_path.is_dir():
        torchvision_dataset_path.mkdir(parents=True)

    if 'train' in not_ready or 'dev' in not_ready:
        train_dataset = FashionMNIST(
            torchvision_dataset_path,
            train=True,
            download=True,
            transform=Compose([Resize((32, 32)),
                               ToTensor(),
                               Lambda(expand)]),
            target_transform=lambda t: torch.tensor(t)
        )
        train_dataset = torchvision_to_tensor(train_dataset)
        dev_dataset, train_dataset = split_tensor_dataset(train_dataset, 0.2)
        torch.save(train_dataset, tensor_datasets_path / 'fmnist-train.pt')
        print('FashionMNIST train tensor dataset ready')

        torch.save(dev_dataset, tensor_datasets_path / 'fmnist-dev.pt')
        print('FashionMNIST dev tensor dataset ready')

    if 'test' in not_ready:
        test_dataset = FashionMNIST(
            torchvision_dataset_path,
            train=False,
            download=True,
            transform=Compose([Resize((32, 32)),
                               ToTensor(),
                               Lambda(expand)]),
            target_transform=lambda t: torch.tensor(t)
        )
        test_dataset = torchvision_to_tensor(test_dataset)
        torch.save(test_dataset, tensor_datasets_path / 'fmnist-test.pt')
        print('FashionMNIST test tensor dataset ready')


def prepare_cifar100(root='data'):
    root = Path(root)
    tensor_datasets_path = root / 'tensor_datasets'
    if not tensor_datasets_path.is_dir():
        tensor_datasets_path.mkdir(parents=True)

    not_ready = []
    for segment in ['train', 'test', 'dev']:
        if (tensor_datasets_path / f'cifar100-{segment}.pt').exists():
            print(f'CIFAR100 {segment} tensor dataset ready')
        else:
            not_ready.append(segment)
    if len(not_ready) == 0:
        return

    torchvision_dataset_path = root / 'torchvision_datasets' / 'cifar100'
    if not torchvision_dataset_path.is_dir():
        torchvision_dataset_path.mkdir(parents=True)

    if 'train' in not_ready or 'dev' in not_ready:
        train_dataset = CIFAR100(
            root=torchvision_dataset_path,
            train=True,
            transform=ToTensor(),
            download=True,
            target_transform=lambda t: torch.tensor(t)
        )
        train_dataset = torchvision_to_tensor(train_dataset)
        dev_dataset, train_dataset = split_tensor_dataset(train_dataset, 0.2)
        torch.save(train_dataset, tensor_datasets_path / 'cifar100-train.pt')
        print('CIFAR100 train tensor dataset ready')

        torch.save(dev_dataset, tensor_datasets_path / 'cifar100-dev.pt')
        print('CIFAR100 dev tensor dataset ready')

    if 'test' in not_ready:
        test_dataset = CIFAR100(
            root=torchvision_dataset_path,
            train=False,
            transform=ToTensor(),
            download=True,
            target_transform=lambda t: torch.tensor(t)
        )
        test_dataset = torchvision_to_tensor(test_dataset)
        torch.save(test_dataset, tensor_datasets_path / 'cifar100-test.pt')
        print(f'CIFAR100 test tensor dataset ready')


def prepare_mnist(root='data'):
    root = Path(root)
    tensor_datasets_path = root / 'tensor_datasets'
    if not tensor_datasets_path.is_dir():
        tensor_datasets_path.mkdir(parents=True)

    not_ready = []
    for segment in ['train', 'test', 'dev']:
        if (tensor_datasets_path / f'mnist-{segment}.pt').exists():
            print(f'MNIST {segment} tensor dataset ready')
        else:
            not_ready.append(segment)
    if len(not_ready) == 0:
        return

    torchvision_dataset_path = root / 'torchvision_datasets' / 'mnist'
    if not torchvision_dataset_path.is_dir():
        torchvision_dataset_path.mkdir(parents=True)

    if 'train' in not_ready or 'dev' in not_ready:
        train_dataset = MNIST(
            torchvision_dataset_path,
            train=True,
            download=True,
            transform=Compose([Resize((32, 32)),
                               ToTensor(),
                               Lambda(expand)]),
            target_transform=lambda t: torch.tensor(t)
        )
        train_dataset = torchvision_to_tensor(train_dataset)
        dev_dataset, train_dataset = split_tensor_dataset(train_dataset, 0.2)
        torch.save(train_dataset, tensor_datasets_path / 'mnist-train.pt')
        print('MNIST train tensor dataset ready')

        torch.save(dev_dataset, tensor_datasets_path / 'mnist-dev.pt')
        print('MNIST dev tensor dataset ready')

    if 'test' in not_ready:
        test_dataset = MNIST(
            torchvision_dataset_path,
            train=False,
            download=True,
            transform=Compose([Resize((32, 32)),
                               ToTensor(),
                               Lambda(expand)]),
            target_transform=lambda t: torch.tensor(t)
        )
        test_dataset = torchvision_to_tensor(test_dataset)
        torch.save(test_dataset, tensor_datasets_path / 'mnist-test.pt')
        print('MNIST test tensor dataset ready')


def prepare_stl10(root='data'):
    root = Path(root)
    tensor_datasets_path = root / 'tensor_datasets'
    if not tensor_datasets_path.is_dir():
        tensor_datasets_path.mkdir(parents=True)

    not_ready = []
    for segment in ['train', 'test', 'dev', 'aux']:
        if (tensor_datasets_path / f'stl10-{segment}.pt').exists():
            print(f'STL10 {segment} tensor dataset ready')
        else:
            not_ready.append(segment)
    if len(not_ready) == 0:
        return

    torchvision_dataset_path = root / 'torchvision_datasets' / 'stl10'
    if not torchvision_dataset_path.is_dir():
        torchvision_dataset_path.mkdir(parents=True)

    if 'train' in not_ready or 'dev' in not_ready:
        train_dataset = STL10(
            torchvision_dataset_path,
            split='train',
            transform=Compose([Resize((32, 32)), ToTensor()]),
            target_transform=lambda t: torch.tensor(t),
            download=True
        )
        train_dataset = torchvision_to_tensor(train_dataset)
        dev_dataset, train_dataset = split_tensor_dataset(train_dataset, 0.2)
        torch.save(train_dataset, tensor_datasets_path / 'stl10-train.pt')
        print('STL10 train tensor dataset ready')
        torch.save(dev_dataset, tensor_datasets_path / 'stl10-dev.pt')
        print('STL10 dev tensor dataset ready')

    if 'test' in not_ready:
        test_dataset = STL10(
            torchvision_dataset_path,
            split='test',
            transform=Compose([Resize((32, 32)), ToTensor()]),
            target_transform=lambda t: torch.tensor(t),
            download=True
        )
        test_dataset = torchvision_to_tensor(test_dataset)
        torch.save(test_dataset, tensor_datasets_path / 'stl10-test.pt')
        print('STL10 test tensor dataset ready')

    if 'aux' in not_ready:
        aux_dataset = STL10(
            torchvision_dataset_path,
            split='unlabeled',
            transform=Compose([Resize((32, 32)), ToTensor()]),
            target_transform=lambda t: torch.tensor(t),
            download=True
        )
        aux_dataset = torchvision_to_tensor(aux_dataset)
        torch.save(aux_dataset, tensor_datasets_path / 'stl10-aux.pt')
        print('STL10 aux tensor dataset ready')


def prepare_cifar10(root='data'):
    root = Path(root)
    tensor_datasets_path = root / 'tensor_datasets'
    if not tensor_datasets_path.is_dir():
        tensor_datasets_path.mkdir(parents=True)

    not_ready = []
    for segment in ['train', 'test', 'dev']:
        if (tensor_datasets_path / f'cifar10-{segment}.pt').exists():
            print(f'CIFAR10 {segment} tensor dataset ready')
        else:
            not_ready.append(segment)
    if len(not_ready) == 0:
        return

    torchvision_dataset_path = root / 'torchvision_datasets' / 'cifar10'
    if not torchvision_dataset_path.is_dir():
        torchvision_dataset_path.mkdir(parents=True)

    if 'train' in not_ready or 'dev' in not_ready:
        train_dataset = CIFAR10(
            root=torchvision_dataset_path,
            train=True,
            transform=ToTensor(),
            download=True,
            target_transform=lambda t: torch.tensor(t)
        )
        train_dataset = torchvision_to_tensor(train_dataset)
        dev_dataset, train_dataset = split_tensor_dataset(train_dataset, 0.2)
        torch.save(train_dataset, tensor_datasets_path / 'cifar10-train.pt')
        print('CIFAR10 train tensor dataset ready')

        torch.save(dev_dataset, tensor_datasets_path / 'cifar10-dev.pt')
        print('CIFAR10 dev tensor dataset ready')

    if 'test' in not_ready:
        test_dataset = CIFAR10(
            root=torchvision_dataset_path,
            train=False,
            transform=ToTensor(),
            download=True,
            target_transform=lambda t: torch.tensor(t)
        )
        test_dataset = torchvision_to_tensor(test_dataset)
        torch.save(test_dataset, tensor_datasets_path / 'cifar10-test.pt')
        print(f'CIFAR10 test tensor dataset ready')


def prepare_svhn(root='data'):
    root = Path(root)
    tensor_datasets_path = root / 'tensor_datasets'
    if not tensor_datasets_path.is_dir():
        tensor_datasets_path.mkdir(parents=True)

    not_ready = []
    for segment in ['train', 'test', 'dev']:
        if (tensor_datasets_path / f'svhn-{segment}.pt').exists():
            print(f'SVHN {segment} tensor dataset ready')
        else:
            not_ready.append(segment)
    if len(not_ready) == 0:
        return

    torchvision_dataset_path = root / 'torchvision_datasets' / 'svhn'
    if not torchvision_dataset_path.is_dir():
        torchvision_dataset_path.mkdir(parents=True)

    if 'train' in not_ready or 'dev' in not_ready:
        train_dataset = SVHN(
            root=torchvision_dataset_path,
            split='train',
            transform=ToTensor(),
            download=True,
            target_transform=lambda t: torch.tensor(t)
        )
        train_dataset = torchvision_to_tensor(train_dataset)
        dev_dataset, train_dataset = split_tensor_dataset(train_dataset, 0.2)
        torch.save(train_dataset, tensor_datasets_path / 'svhn-train.pt')
        print('SVHN train tensor dataset ready')

        torch.save(dev_dataset, tensor_datasets_path / 'svhn-dev.pt')
        print('SVHN dev tensor dataset ready')

    if 'test' in not_ready:
        test_dataset = SVHN(
            root=torchvision_dataset_path,
            split='test',
            transform=ToTensor(),
            download=True,
            target_transform=lambda t: torch.tensor(t)
        )
        test_dataset = torchvision_to_tensor(test_dataset)
        torch.save(test_dataset, tensor_datasets_path / 'svhn-test.pt')
        print(f'SVHN test tensor dataset ready')


def prepare_dataset(dataset_name, root='data'):
    if dataset_name == 'cifar100':
        prepare_cifar100(root)
    elif dataset_name == 'cifar10':
        prepare_cifar10(root)
    elif dataset_name == 'mnist':
        prepare_mnist(root)
    elif dataset_name == 'stl10':
        prepare_stl10(root)
    elif dataset_name == 'kmnist':
        prepare_kmnist(root)
    elif dataset_name == 'fmnist':
        prepare_fmnist(root)
    elif dataset_name == 'svhn':
        prepare_svhn(root)
    else:
        raise ValueError('unrecognized dataset_name')
