from z3.z3 import ArithRef
from dataclasses import dataclass


@dataclass
class Z3Data:
    n_steps: int
    """Number of steps in the simulation"""
    n_features: list[int]
    """Number of features in each layer"""
    n_spikes: dict[tuple[int, int], ArithRef]
    """(layer, neuron) -> n_spikes(ArithRef)"""
    weight: dict[tuple[int, int, int], float]
    """(presynaptic layer, presynaptic neuron, postsynaptic neuron) -> float"""


@dataclass
class Abstraction:
    delta_counts: int
    """Number of delta counts"""
