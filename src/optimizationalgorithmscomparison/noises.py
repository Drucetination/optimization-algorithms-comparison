from enum import Enum


class NoiseType(Enum):
    ADDITIVE = 1,
    MULTIPLICATIVE = 2


class Noise:

    def __init__(self, noise_type, distribution, value):
        self.noise_type = noise_type
        self.distribution = distribution
        self.value = value
