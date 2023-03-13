from lab1.tools import generate_quadratic_func
from pprint import pprint
a = generate_quadratic_func(10, 10)

print(a)
print(a.cond())
pprint(a.get_A())