from lab1.tools import generate_quadratic_func
from lab1.grad import grad_down, grad_down_dichotomy
import sympy as sp
from random import randint
from dataclasses import dataclass


# here also task7
@dataclass
class OutputDTO:
    n: int
    k: int
    const_grag_was_broken: list[bool]
    const_grag_iter: list[int]
    dichotomy_grag_was_broken: list[bool]
    dichotomy_grag_iter: list[int]


def T(n, k):
    start_point = sp.Matrix([[randint(-10, 10) for _ in range(n)]])
    f = generate_quadratic_func(n, k)
    const_grag = grad_down(n, str(f), start_point)
    dichotomy_grag = grad_down_dichotomy(n, str(f), start_point)

    return [const_grag, dichotomy_grag]


# todo graphics to this lists
def stat(out):
    print(f"{out.n = } {out.k = }")
    print(f"{out.const_grag_was_broken=}")
    print(f"{out.dichotomy_grag_was_broken=}")
    print(f"{out.const_grag_iter=}")
    print(f"{out.dichotomy_grag_iter=}")
    const_grag_sr_iter = sum(out.const_grag_iter) / len(out.const_grag_iter)
    dichotomy_grag_sr_iter = sum(out.dichotomy_grag_iter) / len(out.dichotomy_grag_iter)
    print(f"{const_grag_sr_iter=}")
    print(f"{dichotomy_grag_sr_iter=}")
    print("---------------------------------")


for n in range(2, 1000, 10):
    for k in range(1, 1000, 10):
        out = OutputDTO(
            # alpha=[],
            n=n,
            k=k,
            const_grag_iter=[],
            const_grag_was_broken=[],
            dichotomy_grag_iter=[],
            dichotomy_grag_was_broken=[],
        )
        print("waiting", end="")
        for i in range(10):
            print(".", end="")
            const_grag, dichotomy_grag = T(n, k)
            out.const_grag_was_broken.append(const_grag.was_broken)
            out.const_grag_iter.append(len(const_grag.points))
            out.dichotomy_grag_was_broken.append(const_grag.was_broken)
            out.dichotomy_grag_iter.append(len(dichotomy_grag.points))
        print("\r", end="")
        stat(out)
