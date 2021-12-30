from enum import Enum


class NoiseType(Enum):
    ADDITIVE_GAUSSIAN = 1,
    MULTIPLICATIVE_GAUSSIAN = 2


class Noise:

    def __init__(self, noise_type, value):
        self.noise_type = noise_type
        self.value = value
