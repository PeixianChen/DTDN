from __future__ import absolute_import

from .lsr import LSRLoss
from .triplet import TripletLoss
from .neightboor import InvNet
from .neightboor_office import InvNet_office

__all__ = [
    'TripletLoss',
    'LSRLoss',
    'InvNet',
    'InvNet_office', 
]
