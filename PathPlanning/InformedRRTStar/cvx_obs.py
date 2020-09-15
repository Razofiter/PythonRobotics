# Functia cost reprezinta suma acceleratiilor patratice, iar matricile si vectorii de mai jos se formeaza pe baza problemei de control
# Vezi: https://nbviewer.jupyter.org/github/cvxgrp/cvx_short_course/blob/master/intro/control.ipynb

import matplotlib.pyplot as plt
import numpy as np
import cvxpy as cp
import time
#import gurobipy

# Generate data for control problem.
np.random.seed(1)
# Number of states
n = 4
# Number of inputs
m = 2
# Time step
h = 0.2
# Total time
T = 20

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

# Stop state
x_f = np.array([4,8,0,0])

# Initial precomputed path (waypoint based)
wpX = [0, 0.3088583320551013, 0.6177166641102027, 0.926574996165304, 1.2354333282204053, 1.5442916602755066, 1.8531499923306078, 2.162008324385709, 2.4708666564408106, 2.779724988495912, 3.0885833205510136, 3.397441652606115, 3.477974454105484, 3.477974454105484, 3.7088334589965433, 3.9396924638876025, 4.170551468778662, 4.401410473669721, 4.632269478560779, 4.824214323797891, 4.824214323797891, 4.8720239989402785, 4.919833674082666, 4.967643349225053, 5.0154530243674404, 5.063262699509828, 5.0828617226182375, 5.0828617226182375, 5]
wpY = [0, 0.39320036968464406, 0.7864007393692881, 1.1796011090539322, 1.5728014787385762, 1.9660018484232202, 2.3592022181078645, 2.7524025877925085, 3.1456029574771525, 3.5388033271617965, 3.9320036968464405, 4.325204066531085, 4.427728505844715, 4.427728505844715, 4.871241887672941, 5.314755269501167, 5.758268651329393, 6.2017820331576194, 6.645295414985846, 7.014049094210008, 7.014049094210008, 7.511758080428641, 8.009467066647273, 8.507176052865907, 9.00488503908454, 9.502594025303173, 9.706624065103924, 9.706624065103924, 10]
wp = np.matrix([wpX,wpY])

# Obstacle
xmin = 2
xmax = 3
ymin = 5
ymax = 6

print(cp.sum(x_0))
#print(wp.shape)
# Safe distance
l = 2.4

# Arbitrary large constant used similarily as the one in eq. 11 >   http://www.et.byu.edu/~beard/papers/library/RichardsEtAl02.pdf
R = 10000000

# Form and solve control problem.
x = cp.Variable((n, K+1))
u = cp.Variable((m, K))
b = cp.Variable((4, K),boolean=True)
#b = cp.Variable((4, K))
#print(x.shape)


cost = 0
constr = []
for t in range(0,K):
    cost += cp.sum_squares(u[:,t])

    constr += [x[:,t+1] == A@x[:,t] + B@u[:,t],
               x[0,t] - xmin <= R*b[0,t]-l,
               x[1,t] - ymin <= R*b[1,t]-l,
               xmax - x[0,t] <= R*b[2,t]-l,
               ymax - x[1,t] <= R*b[3,t]-l,
               cp.sum(b[:,t]) <= 3,
               cp.norm(u[:,t], 2) <= Amax,
               cp.norm(x[2:,t], 2) <= Vmax]
    
# sums problem objectives and concatenates constraints with the initial and final states.
constr += [x[:,K] == x_f, x[:,0] == x_0]
problem = cp.Problem(cp.Minimize(cost), constr)

# Time stamp for problem start execution
start = time.time()

problem.solve(verbose=True,solver=cp.MOSEK)

# Print execution time
end = time.time()
print(end - start)

#print(b.value)


# Plot results.
f = plt.figure(1)

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
#print(x1)
plt.plot(x1)
plt.ylabel(r"$p_x$", fontsize=16)
plt.yticks([-10, 0, 10])
plt.xticks([])

# Plot (x_t)_2.
plt.subplot(6,1,4)
x2 = x[1,:].value
#print(x2)
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

plt.figure(2)
#rrt.draw_graph()
# Reprezentam grafic punctele determinate folosind metoda IRRT*
#plt.plot([x for (x, y) in wp], [y for (x, y) in wp], '-r')
plt.plot(wpX, wpY, '-r')
# Reprezentam grafic punctele GPS determinate, functie de constrangerile generate de scenariul de zbor
#plt.plot([x for (x, y) in wp], [y for (x, y) in wp], 'Hb')
rectangle = plt.Rectangle((xmin,ymin), xmax-xmin, ymax-ymin, fc='blue', ec="red")
plt.gca().add_patch(rectangle)
# Reprezentam grafic punctele GPS obtinute dupa aplicarea algoritmului de optimizare a traciectoriei
plt.plot(x1, x2, '*y')
plt.plot(x_0[0], x_0[1], "xr")
plt.plot(x_f[0], x_f[1], "xr")
plt.xlabel(r"$x[m]$", fontsize=16)
plt.ylabel(r"$y[m]$", fontsize=16)
plt.grid(True)
plt.axis('scaled')
plt.show()
