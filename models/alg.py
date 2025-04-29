# coding=utf-8
from models.algs.base import Base
from models.algs.base_joint import Base_joint
from models.algs.ordinalReg import OrdinalReg
from models.algs.ordinalReg_joint import OrdinalReg_joint
from models.algs.aggdNet import AggdNet
from models.algs.aggdNet_joint import AggdNet_joint
from models.algs.ornet import ORNet
from models.algs.ornet_joint import ORNet_joint


ALGORITHMS = [
    'Base',
    'OrdinalReg',
    'AggdNet',
    'ORNet',
    'Base_joint',
    'OrdinalReg_joint',
    'AggdNet_joint',
    'ORNet_joint'
]


def get_algorithm_class(algorithm_name):
    if algorithm_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(algorithm_name))
    return globals()[algorithm_name]
