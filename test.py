from sympy import symbols, Eq, solve

# Define the variables
x, y = symbols('x y')
r1, r2 = symbols('r1 r2')
g0, g1 = symbols('g0 g1')

# Define the system of equations
equation1 = Eq(x + y, g0)
equation2 = Eq(x*1/r1 + y/r2, g1)

# Solve the system of equations
solution = solve((equation1, equation2), (x, y))

# Print the solution
print("Solution:", solution)

phi1 = symbols('phi1')
phi2 = symbols('phi2')

# print(solution.keys())
print(solution[x].subs({g0: 1, g1: phi1/(1-phi1)}).simplify())
print(solution[y].subs({g0: 1, g1: phi1/(1-phi1)}).simplify())