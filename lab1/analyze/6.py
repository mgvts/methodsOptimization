import json
from dataclasses import dataclass
from random import uniform, randint
from time import time

from lab1.fast_grad import grad_down, grad_down_dichotomy, grad_down_wolfe
from lab1.tools import fast_generate_quadratic_func


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


class Saver:
    def __init__(self):
        self.data = {}

    def add(self, n: int, k: int, const_grag_sr_iter: float,
            dichotomy_grag_sr_iter: float,
            wolfe_grag_sr_iter: float,
            const_grag_was_broken: int,
            dichotomy_grag_was_broken: int,
            wolfe_grag_was_broken: int,
            ):
        self.data.update({
            'n': n,
            'k': k,
            'const_grag_sr_iter': const_grag_sr_iter,
            'dichotomy_grag_sr_iter': dichotomy_grag_sr_iter,
            'wolfe_grag_sr_iter': wolfe_grag_sr_iter,
            'const_grag_was_broken': const_grag_was_broken,
            'dichotomy_grag_was_broken': dichotomy_grag_was_broken,
            'wolfe_grag_was_broken': wolfe_grag_was_broken
        })

        try:
            r = json.dumps(self.data)
            with open('../data/count-iter67.json', 'w') as file:
                file.write(r)
        except Exception:
            self.add(n, k, -1, -1, -1, 5, 5, 5)


data = Saver()


def T(n, k, f):
    const_grag = []
    dichotomy_grag = []
    wolfe_grad = []

    const = 0
    dich = 0
    wolfe = 0
    const_c = 0
    dich_c = 0
    wolfe_c = 0
    for i in range(5):
        start_point = [randint(-100, 100) for _ in range(n)]
        constDto = grad_down(f, start_point, alpha=1 / (80 * k + 1))
        dichDto = grad_down_dichotomy(f, start_point)
        wolfeDto = grad_down_wolfe(f, start_point)

        const_grag.append(len(constDto.points))
        dichotomy_grag.append(len(dichDto.points))
        wolfe_grad.append(len(wolfeDto.points))

        if constDto.was_broken:
            const += 1
        else:
            const_c += 1
        if dichDto.was_broken:
            dich += 1
        else:
            dich_c += 1
        if wolfeDto.was_broken:
            wolfe += 1
        else:
            wolfe_c += 1

    return [(sum(const_grag) / const_c if const_c != 0 else -1, const),
            (sum(dichotomy_grag) / dich_c if dich_c != 0 else -1, dich),
            (sum(wolfe_grad) / wolfe_c if wolfe_c != 0 else -1, wolfe)
            ]


def stat(out):
    print(f"{out.n = } {out.k = }")
    print(
        f"const_grag_was_broken False:{out.const_grag_was_broken}")
    print(
        f"dichotomy_grag_was_broken False:{out.dichotomy_grag_was_broken}")
    print(
        f"dichotomy_grag_was_broken False:{out.wolfe_grag_was_broken}")

    const_grag_sr_iter = sum(out.const_grag_iter) / len(out.const_grag_iter) if -1 not in out.const_grag_iter else -1
    dichotomy_grag_sr_iter = sum(out.dichotomy_grag_iter) / len(
        out.dichotomy_grag_iter) if -1 not in out.dichotomy_grag_iter else -1
    wolfe_grag_sr_iter = sum(out.wolfe_grag_iter) / len(out.wolfe_grag_iter) if -1 not in out.wolfe_grag_iter else -1

    data.add(n, k, const_grag_sr_iter, dichotomy_grag_sr_iter, wolfe_grag_sr_iter, out.const_grag_was_broken,
             out.dichotomy_grag_was_broken, out.wolfe_grag_was_broken)

    print(f"{const_grag_sr_iter=}")
    print(f"{dichotomy_grag_sr_iter=}")
    print(f"{wolfe_grag_sr_iter=}")

    print("---------------------------------")


for n in range(2, 1000, 10):
    for k in range(1, 1000, 10):
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
        for i in range(5):
            print(".", end="")
            f = fast_generate_quadratic_func(n, k)
            const_grag, dichotomy_grag, wolfe_grad = T(n, k, f)
            out.const_grag_iter.append(const_grag[0])
            out.const_grag_was_broken = const_grag[1]
            out.dichotomy_grag_iter.append(dichotomy_grag[0])
            out.dichotomy_grag_was_broken = dichotomy_grag[1]
            out.wolfe_grag_iter.append(wolfe_grad[0])
            out.wolfe_grag_was_broken = dichotomy_grag[1]
        delta = st - time()
        print()
        print("\r", end="")
        stat(out)
