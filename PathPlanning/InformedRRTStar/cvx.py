
import time

import numpy as np
import cvxpy as cp
import math

# Generate data for control problem.
np.random.seed(1)
# Number of states
n = 4
# Number of inputs
m = 2
# Time step
h = 0.2
# Total time
T = 5.8

#Number of iterations
K = int(T/h)

# Vehicle related limits
Amax = 2 #m/s^2
Vmax = 12 #m/s

# Kinematics of a point mass
A = np.matrix([[1,0,h,0],[0,1,0,h],[0,0,1,0],[0,0,0,1]])

B = np.matrix([[(h**2)/2,0],[0,(h**2)/2],[h,0],[0,h]])

# Start state
x_0 = np.array([0,0,0,0])

# Initial precomputed path (waypoint based)
wp = np.matrix([[0, 0.3088583320551013, 0.6177166641102027, 0.926574996165304, 1.2354333282204053, 1.5442916602755066, 1.8531499923306078, 2.162008324385709, 2.4708666564408106, 2.779724988495912, 3.0885833205510136, 3.397441652606115, 3.477974454105484, 3.477974454105484, 3.7088334589965433, 3.9396924638876025, 4.170551468778662, 4.401410473669721, 4.632269478560779, 4.824214323797891, 4.824214323797891, 4.8720239989402785, 4.919833674082666, 4.967643349225053, 5.0154530243674404, 5.063262699509828, 5.0828617226182375, 5.0828617226182375, 5],
                [0, 0.39320036968464406, 0.7864007393692881, 1.1796011090539322, 1.5728014787385762, 1.9660018484232202, 2.3592022181078645, 2.7524025877925085, 3.1456029574771525, 3.5388033271617965, 3.9320036968464405, 4.325204066531085, 4.427728505844715, 4.427728505844715, 4.871241887672941, 5.314755269501167, 5.758268651329393, 6.2017820331576194, 6.645295414985846, 7.014049094210008, 7.014049094210008, 7.511758080428641, 8.009467066647273, 8.507176052865907, 9.00488503908454, 9.502594025303173, 9.706624065103924, 9.706624065103924, 10]])

#print(wp.shape)
# Safe distance
l = 2.4

# Form and solve control problem.

x = cp.Variable((n, K+1))
u = cp.Variable((m, K))
#print(x.shape)


cost = 0
constr = []
for t in range(0,K):
    cost += cp.sum_squares(u[:,t])

    #diff = wp[:2,t] - x[:2,t][:,None]
    #print(diff)
    constr += [x[:,t+1] == A@x[:,t] + B@u[:,t],
               #cp.norm(wp[:,t] - x[:2,t][:,None],'inf') <= l,
               cp.norm(x[:2,t][:,None] - wp[:,t],2) <= l,
               cp.norm(u[:,t], 2) <= Amax,
               cp.norm(x[2:,t], 2) <= Vmax]
# sums problem objectives and concatenates constraints with the initial and final states.
constr += [x[:,K] == np.array([5,10,0,0]), x[:,0] == x_0]
problem = cp.Problem(cp.Minimize(cost), constr)
#problem.solve('qcp=True')

# Time stamp for problem start execution
start = time.time()

problem.solve()

# Print execution time
end = time.time()
print(end - start)

# Plot results.
import matplotlib.pyplot as plt
#from IPython import get_ipython
#get_ipython().run_line_magic('matplotlib', 'inline')
#get_ipython().run_line_magic('config', 'InlineBackend.figure_format = "svg"')
#%matplotlib inline
#%config InlineBackend.figure_format = 'svg'

f = plt.figure()

# Plot (u_t)_1.
ax = f.add_subplot(611)
plt.plot(u[0,:].value)
plt.ylabel(r"$a_x$", fontsize=16)
plt.yticks(np.linspace(-1, 1, 3))
plt.xticks([])
plt.grid(True)

# Plot (u_t)_2.
plt.subplot(6,1,2)
plt.plot(u[1,:].value)
plt.ylabel(r"$a_y$", fontsize=16)
plt.yticks(np.linspace(-1, 1, 3))
plt.xticks([])

# Plot (x_t)_1.
plt.subplot(6,1,3)
x1 = x[0,:].value
print(x1)
plt.plot(x1)
plt.ylabel(r"$p_x$", fontsize=16)
plt.yticks([-10, 0, 10])
plt.xticks([])

# Plot (x_t)_2.
plt.subplot(6,1,4)
x2 = x[1,:].value
print(x2)
plt.plot(x2)
plt.yticks([-12, 0, 12])
plt.ylabel(r"$p_y$", fontsize=16)

# Plot (x_t)_3.
plt.subplot(6,1,5)
x3 = x[2,:].value
#print(x3)
plt.plot(x3)
plt.ylabel(r"$v_x$", fontsize=16)
plt.ylim([-Vmax, Vmax])
plt.xticks([])

# Plot (x_t)_4.
plt.subplot(6,1,6)
x4 = x[3,:].value
#print(x4)
plt.plot(range(K+1), x4)
plt.ylabel(r"$v_y$", fontsize=16)
plt.ylim([-Vmax, Vmax])
plt.xticks([])
plt.xlabel(r"$t$", fontsize=16)

plt.tight_layout()
plt.show()
