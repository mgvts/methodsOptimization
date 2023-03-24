from lab1.tools import fast_generate_quadratic_func

from lab1.fast_grad import grad_down

f = fast_generate_quadratic_func(2, 10)
x = grad_down(f, [10, 10])

print(x)
