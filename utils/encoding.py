from z3.z3 import Int, Solver, And, Implies, sat, ArithRef
from uuid import uuid4


def floor(in_: ArithRef, floor_: ArithRef):
    return And(floor_ <= in_, in_ < floor_ + 1)
