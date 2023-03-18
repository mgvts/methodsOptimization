from lab1.tools import generate_quadratic_func
from lab1.grad import grad_down, grad_down_dichotomy, grad_down_wolfe
import sympy as sp
from random import randint
from dataclasses import dataclass

from time import time


# here also task7
@dataclass
class OutputDTO:
    n: int
    k: int
    const_grag_was_broken: list[bool]
    const_grag_iter: list[int]
    dichotomy_grag_was_broken: list[bool]
    dichotomy_grag_iter: list[int]
    wolfe_grag_was_broken: list[bool]
    wolfe_grag_iter: list[int]


def T(n, k):
    start_point = sp.Matrix([[randint(-10, 10) for _ in range(n)]])
    f = generate_quadratic_func(n, k)
    const_grag = grad_down(n, str(f), start_point)
    dichotomy_grag = grad_down_dichotomy(n, str(f), start_point)
    wolfe_grad = grad_down_wolfe(n, str(f), start_point)

    return [const_grag, dichotomy_grag, wolfe_grad]


# todo graphics to this lists
def stat(out):
    print(f"{out.n = } {out.k = }")
    print(
        f"const_grag_was_broken False:{out.const_grag_was_broken.count(False)}"
        f" True:{out.const_grag_was_broken.count(True)}")
    print(
        f"dichotomy_grag_was_broken False:{out.dichotomy_grag_was_broken.count(False)}"
        f" True:{out.dichotomy_grag_was_broken.count(True)}")
    print(
        f"dichotomy_grag_was_broken False:{out.wolfe_grag_was_broken.count(False)}"
        f" True:{out.wolfe_grag_was_broken.count(True)}")
    # print(f"{out.const_grag_iter=}")
    # print(f"{out.dichotomy_grag_iter=}")
    const_grag_sr_iter = sum(out.const_grag_iter) / len(out.const_grag_iter)
    dichotomy_grag_sr_iter = sum(out.dichotomy_grag_iter) / len(out.dichotomy_grag_iter)
    wolfe_grag_sr_iter = sum(out.dichotomy_grag_iter) / len(out.dichotomy_grag_iter)

    print(f"{const_grag_sr_iter=}")
    print(f"{dichotomy_grag_sr_iter=}")
    print(f"{wolfe_grag_sr_iter=}")

    print("---------------------------------")


for n in range(2, 1000, 10):
    for k in [1, 2, 10, 100]:
        # for k in range(1, 1000, 10):
        out = OutputDTO(
            n=n,
            k=k,
            const_grag_iter=[],
            const_grag_was_broken=[],
            dichotomy_grag_iter=[],
            dichotomy_grag_was_broken=[],
            wolfe_grag_iter=[],
            wolfe_grag_was_broken=[]
        )
        print("waiting", end="")
        st = time()
        for i in range(10):
            # try:
            print(".", end="")
            const_grag, dichotomy_grag, wolfe_grad = T(n, k)
            out.const_grag_was_broken.append(const_grag.was_broken)
            out.const_grag_iter.append(len(const_grag.points))
            out.dichotomy_grag_was_broken.append(const_grag.was_broken)
            out.dichotomy_grag_iter.append(len(dichotomy_grag.points))
            out.wolfe_grag_iter.append(len(wolfe_grad.points))
            out.wolfe_grag_was_broken.append(wolfe_grad.was_broken)
            # except Exception:
            #     print("exp", end="")
            #     pass
        delta = st - time()
        print()
        # print("\r", end="")
        stat(out)
