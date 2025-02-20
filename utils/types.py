from z3.z3 import ArithRef
from dataclasses import dataclass


@dataclass
class Z3Data:
    n_spikes: dict[tuple[int, int, int], ArithRef]
    """(layer, step, neuron) -> n_spikes(ArithRef)"""
