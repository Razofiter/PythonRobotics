import cvxpy as cp

x = cp.Variable(name='x')
y = cp.Variable(name='y')
expr = x
const1 = x == 3
const2 = y * y <= 5
prob = cp.Problem(cp.Minimize(expr), [const1, const2])
prob.solve()
