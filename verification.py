from z3 import Int, And, Or, Not, Implies, Solver, sat

if __name__ == "__main__":
    x = Int("x")
    y = Int("y")
    z = Int("z")

    s = Solver()
    s.add(And(x > 0, y > 0, z > 0))
    s.add(Implies(x > y, x > z))
    s.add(Implies(y > z, y > x))
    s.add(Implies(z > x, z > y))

    if s.check() == sat:
        print(s.model())
    else:
        print("unsat")
