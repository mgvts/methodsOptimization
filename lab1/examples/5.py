from lab1.tools import fast_generate_quadratic_func
from pprint import pprint
a = fast_generate_quadratic_func(1000, 10)

print(a)
print(a.cond())
pprint(a.get_A())