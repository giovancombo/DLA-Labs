""" This module contains the Action enum """
from enum import Enum


class Action(Enum):
    # pylint: disable=invalid-name
    linear_shift: float
    angular_shift: float
    # pylint: enable=invalid-name

    def __new__(cls,
                value: int,
                linear_shift: float = 0,
                angular_shift: float = 0):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.linear_shift = linear_shift
        obj.angular_shift = angular_shift
        return obj

    FORWARD = (0, 0.2, 0)         # (idx azione, linear shift, angular shift)
    ROTATE_RIGHT = (1, 0.1, 0.2)    # originale: (1, 0.1, 0.2) --> rotazione implica leggero spostamento in avanti (linear shift)
    ROTATE_LEFT = (2, 0.1, -0.2)    # originale: (2, 0.1, -0.2) -> rotazione implica leggero spostamento in avanti (linear shift)
