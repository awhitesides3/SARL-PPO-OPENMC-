# a = 3
# print([1.0] * a)

nx = 3
bounds = {f'x{i+1}': ['float', 0.01, 10.0] for i in range(nx)}
print(bounds)
