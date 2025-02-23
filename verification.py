from z3.z3 import Int, Real, Solver, And, Implies, sat
from utils.encoding.encoding import generate_snn

if __name__ == "__main__":
    x = Real("x")
    y = Int("y")
    z = Int("z")

    s = Solver()
    s.add(x == 0.5)
    s.add(floor(x, y))

    if s.check() == sat:
        print(s.model())
    else:
        print("unsat")
