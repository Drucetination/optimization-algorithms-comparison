from enum import Enum


class Metrics(Enum):
    TIME = 1,
    GRADIENT_NORM = 2,
    FUNCTION_VALUE = 3
    # RADIUS = 2, #todo
    # TIME_AND_RADIUS = 3, #todo


