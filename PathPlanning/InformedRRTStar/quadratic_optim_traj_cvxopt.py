import time
import random
import numpy as np
import cvxopt as co
from cvxopt.modeling import variable, op
import math
import copy
import matplotlib.pyplot as plt
from cvxpy.atoms.affine.unary_operators import NegExpression as ne

show_animation = True
ALT_HOLD_CONTROLLER_REFRESH_RATE = 0.02 #[50Hz]

class GraphRepresentation():

    def __init__(self, start, goal,
                 obstacleList, randArea,
                 expandDis=0.5, goalSampleRate=-1, maxIter=100):

        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.minrand = randArea[0]
        self.maxrandX = randArea[1]
        self.maxrandY = randArea[2]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList
        # Time step iteration
        self.h = 0.2
        # Calculam care este raza minima a obstacolelor existente
        obstacleMinRadius = min([min(tpl) for tpl in obstacleList])
        
        '''Vehicle related limits'''
        # Acum determinam viteza maxima cunoscand timpul de iteratie si distanta minimima pe care permitem ca UAV-ul sa o parcurga (raportata la raza minima a obiectelor existente pe harta)
        self.velMax = obstacleMinRadius/0.2
        # Determinam acceleratia maxima
        self.accMax = self.velMax/self.h
        # Calculam distanta pe care o poate parcurge UAV-ul, functie de viteza maxima si de viteza de refresh a algoritmului controlerului care se ocupa cu functia de ALT_HOLD
        self.safeDist = self.velMax*ALT_HOLD_CONTROLLER_REFRESH_RATE
        # In final, calculam distanta (euclidiana) minima fata de obstacolele existente, la care putem esantiona puncte pentru a calcula un traseu apriori
        self.minDist = obstacleMinRadius + self.safeDist

    def OptimizeTraj(self):

        # Number of states 
        n = 4
        # Number of inputs
        m = 2
        
        # Find the euclidean distance between start and goal positions
        dist = math.sqrt((self.start.x - self.goal.x)**2 + (self.start.y - self.goal.y)**2)
        print(dist)
        
        # Minimum time to reach destination if flying in a straight line
        tMin = dist/self.velMax
        print(tMin)
        
        # For the optimization purpose we select longer time
        t = int(tMin)*10

        # Number of iterations
        K = round(t/self.h)
        print(K)

        # Kinematics of a point mass
        A = np.matrix([[1,0,self.h,0],[0,1,0,self.h],[0,0,1,0],[0,0,0,1]])

        B = np.matrix([[(self.h**2)/2,0],[0,(self.h**2)/2],[self.h,0],[0,self.h]])

        # Start state
        x_0 = np.array([0,0,0,0])

        # Form and solve control problem.

        x = co.normal(n, K+1)
        u = co.normal(m, K)

        cost = 0
        #constr = []
        print(x)
        # Convert obstacle list into a matrix
        obstacleList = np.asarray(self.obstacleList)
        print(obstacleList)
        print(self.minDist)
        
        for t in range(0,K):
            cost += u[0,t]**2+u[1,t]**2
           
            #constr.append(x[:,t+1] == A@x[:,t] + B@u[:,t])
            #constr += [(x[:,t+1] == A@x[:,t] + B@u[:,t]),(-(x[0,t]-obstacleList[1,0])**2 - (x[1,t]-obstacleList[1,1])**2 + obstacleList[1,2]**2 <= 0),(math.sqrt(u[0,t]**2 + u[1,t]**2) <= self.accMax),(math.sqrt(x[2,t]**2 + x[3,t]**2) <= self.velMax)]         
        # sums problem objectives and concatenates constraints with the initial and final states.
        constr = u[0,0] <= self.accMax
        #constr += (x[:,K] == np.array([self.goal.x,self.goal.y,1,1])), (x[:,0] == x_0)
        print(constr)
        # Time taken until this point
        #end = time.time()
        #print('Problem formulation:',end - start)
        problem = op(cost, constr)

        problem.solve()
        print(x,u)
        stateMat = np.matrix([x[0,:].value,x[1,:].value,x[2,:].value,x[3,:].value])
        ctrlMat = np.matrix([u[0,:].value,u[1,:].value])
        #print(ctrlMat)
        # Returnam matricea cu starile pentru a extrage din aceasta pozitia (x,y), in vederea reprezentarii grafice
        return stateMat
     
    def drawGraph(self):
        plt.clf()
        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.goal.x, self.goal.y, "xr")
        plt.axis([0, 70, 0, 30])
        plt.grid(True)
        plt.pause(0.01)

class Node():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def main():
    print("Start informed rrt star planning")

    # create obstacles
    obstacleList = [
        (5, 5, 0.5),
        (9, 6, 1),
        (7, 5, 1),
        (2, 2, 1),
        (3, 6, 1),
        (7, 9, 1)
    ]

    # Set params
    rrt = GraphRepresentation(start=[1, 1], goal=[4.5, 4.5],
                          randArea=[0, 70, 30], obstacleList=obstacleList)
    
    optimizedPath = rrt.OptimizeTraj()
    
    #rrt1 = InformedRRTStar(start=[0, 0], goal=[1, 3],
                          #randArea=[-2, 15], obstacleList=obstacleList)
    
    #optimizedPath1 = rrt1.OptimizeTraj()
    print(optimizedPath)
    print(optimizedPath[0,],optimizedPath[1,])

    #print(waypointPath)

    # Plot path
    if show_animation:
        rrt.drawGraph()
        # Reprezentam grafic punctele GPS obtinute dupa aplicarea algoritmului de optimizare a traciectoriei
        plt.plot(optimizedPath[0,], optimizedPath[1,], '*y')
        #plt.plot(optimizedPath1[0,], optimizedPath1[1,], '*y')
        plt.grid(True)
        plt.pause(0.01)
        plt.show()


if __name__ == '__main__':
    main()
