import pandas as pd

df = pd.DataFrame({'n': [1, 2, 1, 2],
                   'iters': [i for i in range(4)],
                   'name': ['a' for i in range(4)]})

# print(df.groupby(['n', 'name']).apply(lambda x: x))
print(df.loc(1).obj)

