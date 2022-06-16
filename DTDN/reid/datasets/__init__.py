from __future__ import absolute_import
from .duke import Duke
from .market import Market
# from .sysu import SYSU
# from .cifar10 import CIFAR10
# from .stl10 import STL10
# from .svhn import SVHN
# from .mnist import MNIST


__factory = {
    'market': Market,
    'duke': Duke,
    # 'sysu': SYSU
    # 'cifar10': cifar10, 
    # 'stl10': stl10, 
    # 'mnist': mnist,
    # 'svhn': svhn
}


def names():
    return sorted(__factory.keys())


def create(name, root, *args, **kwargs):
    """
    Create a dataset instance.

    Parameters
    ----------
    name : str
        The dataset name. Can be one of 'market', 'duke'.
    root : str
        The path to the dataset directory.
    """
    if name not in __factory:
        raise KeyError("Unknown dataset:", name)
    return __factory[name](root, *args, **kwargs)