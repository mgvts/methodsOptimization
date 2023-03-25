import json
from dataclasses import dataclass
from random import randint

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
        self.data = []

    def add(self, n: int, k: int, const_grag_sr_iter: list[float],
            dichotomy_grag_sr_iter: list[float],
            wolfe_grag_sr_iter: list[float],
            const_grag_was_broken: list[int],
            dichotomy_grag_was_broken: list[int],
            wolfe_grag_was_broken: list[int],
            ):
        self.data.append({
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
            self.add(n, k, [], [], [], [], [], [])


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
    for start_point in [[-100 for i in n], [100 for i in n]]:

        a = 1
        if k > 100:
            a = 1.4
        if k > 200:
            a = 2
        if k > 300:
            a = 3
        if k > 500:
            a = 3.3
        if k > 600:
            a = 4
        if k > 800:
            a = 5
        if k > 900:
            a = 6

        constDto = grad_down(f, start_point, alpha=1 / (80 * a * k + 1), max_inter=100)
        dichDto = grad_down_dichotomy(f, start_point)
        wolfeDto = grad_down_wolfe(f, start_point)


        if constDto.was_broken:
            const += 1
        else:
            const_grag.append(len(constDto.points))
            const_c += 1
        if dichDto.was_broken:
            dich += 1
        else:
            dichotomy_grag.append(len(dichDto.points))
            dich_c += 1

        if wolfeDto.was_broken:
            wolfe += 1
        else:
            wolfe_grad.append(len(wolfeDto.points))
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
        f"wolfe_grag_was_broken False:{out.wolfe_grag_was_broken}")

    const_grag_sr_iter = out.const_grag_iter
    dichotomy_grag_sr_iter = out.dichotomy_grag_iter
    wolfe_grag_sr_iter = out.wolfe_grag_iter

    data.add(n, k,
             out.const_grag_iter,
             out.dichotomy_grag_iter,
             out.wolfe_grag_iter,
             out.const_grag_was_broken,
             out.dichotomy_grag_was_broken,
             out.wolfe_grag_was_broken)

    print(f"{const_grag_sr_iter=}")
    print(f"{dichotomy_grag_sr_iter=}")
    print(f"{wolfe_grag_sr_iter=}")

    print("---------------------------------")


for n in range(102, 1000, 100):
    for k in range(2, 1000, 100):
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
        for i in range(5):
            print(".", end="")
            f = fast_generate_quadratic_func(n, k)
            const_grag, dichotomy_grag, wolfe_grad = T(n, k, f)
            out.const_grag_iter.append(const_grag[0])
            out.const_grag_was_broken.append(const_grag[1])
            out.dichotomy_grag_iter.append(dichotomy_grag[0])
            out.dichotomy_grag_was_broken.append(dichotomy_grag[1])
            out.wolfe_grag_iter.append(wolfe_grad[0])
            out.wolfe_grag_was_broken.append(wolfe_grad[1])
        print()
        print("\r", end="")
        stat(out)
