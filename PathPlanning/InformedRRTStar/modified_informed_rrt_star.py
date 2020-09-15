import time
import random
import numpy as np
import cvxpy as cp
import math
import copy
import matplotlib.pyplot as plt

show_animation = True

# Pentru moment nu folosim cele 2 variabile de mai jos. Sunt doar cu titlu informativ asupra modului in care a fost realizat pilotul automat pentru quad
CONTROLLER_INNER_LOOP_REFRESH_RATE = 0.004 # [s]
CONTROLLER_OUTER_LOOP_REFRESH_RATE = 0.8 #[s]

class InformedRRTStar():

    def __init__(self, start, goal,
                 obstacleList, randArea,
                 expandDis=0.5, goalSampleRate=5, maxIter=200):

        self.start = Node(start[0], start[1])
        self.goal = Node(goal[0], goal[1])
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList
        # Time step iteration
        self.h = CONTROLLER_OUTER_LOOP_REFRESH_RATE

        # Calculam care este raza minima a obstacolelor existente
        obstacleMinRadius = min([min(tpl) for tpl in obstacleList])/2
        print("Raza minima: %f" % obstacleMinRadius)
        
        '''Vehicle related limits'''
        # Acum determinam viteza maxima cunoscand timpul de iteratie si distanta minimima pe care permitem ca UAV-ul sa o parcurga (raportata la raza minima a obiectelor existente pe harta)
        self.velMax = obstacleMinRadius/self.h
        print("Viteza maxima: %f" % self.velMax)
        # Impunem acceleratia de translatie maxima, raportat la dinx`amica UAVului
        self.accMax = 2   # m/s^2
        #self.accMax = (2*obstacleMinRadius)/self.h**2
        print("Acceleratia maxima: %f" % self.accMax)
        # Calculam distanta pe care o poate parcurge UAV-ul, functie de viteza maxima si de viteza de refresh a algoritmului controlerului care se ocupa cu functia de ALT_HOLD
        #self.safeDist = self.velMax*self.h
        # In final, calculam distanta (euclidiana) minima fata de obstacolele existente, la care putem esantiona puncte pentru a calcula un traseu apriori
        self.minDist = obstacleMinRadius
        self.expandDis = obstacleMinRadius
        print("Distanta de siguranta: %f" % self.minDist)

    def InformedRRTStarSearch(self, animation=True):

        self.nodeList = [self.start]
        # max length we expect to find in our 'informed' sample space, starts as infinite
        cBest = float('inf')
        pathLen = float('inf')
        solutionSet = set()
        path = None

        # Computing the sampling space
        cMin = math.sqrt(pow(self.start.x - self.goal.x, 2)
                         + pow(self.start.y - self.goal.y, 2))
        xCenter = np.array([[(self.start.x + self.goal.x) / 2.0],
                            [(self.start.y + self.goal.y) / 2.0], [0]])
        a1 = np.array([[(self.goal.x - self.start.x) / cMin],
                       [(self.goal.y - self.start.y) / cMin], [0]])

        etheta = math.atan2(a1[1], a1[0])
        # first column of idenity matrix transposed
        id1_t = np.array([1.0, 0.0, 0.0]).reshape(1, 3)
        M = a1 @ id1_t
        U, S, Vh = np.linalg.svd(M, 1, 1)
        C = np.dot(np.dot(U, np.diag(
            [1.0, 1.0, np.linalg.det(U) * np.linalg.det(np.transpose(Vh))])), Vh)

        for i in range(self.maxIter):
            # Sample space is defined by cBest
            # cMin is the minimum distance between the start point and the goal
            # xCenter is the midpoint between the start and the goal
            # cBest changes when a new path is found
            
            rnd = self.informed_sample(cBest, cMin, xCenter, C)
            nind = self.getNearestListIndex(self.nodeList, rnd)
            nearestNode = self.nodeList[nind]
            # steer
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)
            newNode = self.getNewNode(theta, nind, nearestNode)
            d = self.lineCost(nearestNode, newNode)

            isCollision = self.__CollisionCheck(newNode, self.obstacleList)
            isCollisionEx = self.check_collision_extend(nearestNode, theta, d)

            if isCollision and isCollisionEx:
                nearInds = self.findNearNodes(newNode)
                newNode = self.chooseParent(newNode, nearInds)

                self.nodeList.append(newNode)
                self.rewire(newNode, nearInds)

                if self.isNearGoal(newNode):
                    solutionSet.add(newNode)
                    lastIndex = len(self.nodeList) - 1
                    tempPath = self.getFinalCourse(lastIndex)
                    tempPathLen = self.getPathLen(tempPath)
                    if tempPathLen < pathLen:
                        path = tempPath
                        cBest = tempPathLen
                        print(cBest)
                        # Reconsideram acceleratia de translatie maxima, raportat la lungimea traiectoriei
                        # self.temp_accMax = (2*cBest)/self.h**2
                        # self.accMax = min(self.accMax,self.temp_accMax)   # m/s^2
                        # print("New acc: %2.2f" % self.accMax)
            # Plot the last step of iteration
            if animation and i == self.maxIter-1:
            #if animation:
                self.drawGraph(xCenter=xCenter,
                               cBest=cBest, cMin=cMin,
                               etheta=etheta, rnd=rnd)

        return path

    def chooseParent(self, newNode, nearInds):
        if len(nearInds) == 0:
            return newNode

        dList = []
        for i in nearInds:
            dx = newNode.x - self.nodeList[i].x
            dy = newNode.y - self.nodeList[i].y
            d = math.sqrt(dx ** 2 + dy ** 2)
            theta = math.atan2(dy, dx)
            if self.check_collision_extend(self.nodeList[i], theta, d):
                dList.append(self.nodeList[i].cost + d)
            else:
                dList.append(float('inf'))

        minCost = min(dList)
        minInd = nearInds[dList.index(minCost)]

        if minCost == float('inf'):
            print("mincost is inf")
            return newNode

        newNode.cost = minCost
        newNode.parent = minInd

        return newNode

    def findNearNodes(self, newNode):
        nnode = len(self.nodeList)
        r = 10.0 * math.sqrt((math.log(nnode) / nnode))
        dlist = [(node.x - newNode.x) ** 2
                 + (node.y - newNode.y) ** 2 for node in self.nodeList]
        nearinds = [dlist.index(i) for i in dlist if i <= r ** 2]
        return nearinds

    def informed_sample(self, cMax, cMin, xCenter, C):
        if cMax < float('inf'):
            r = [cMax / 2.0,
                 math.sqrt(cMax**2 - cMin**2) / 2.0,
                 math.sqrt(cMax**2 - cMin**2) / 2.0]
            L = np.diag(r)
            xBall = self.sampleUnitBall()
            rnd = np.dot(np.dot(C, L), xBall) + xCenter
            rnd = [rnd[(0, 0)], rnd[(1, 0)]]
        else:
            rnd = self.sampleFreeSpace()

        return rnd

    def sampleUnitBall(self):
        a = random.random()
        b = random.random()

        if b < a:
            a, b = b, a

        sample = (b * math.cos(2 * math.pi * a / b),
                  b * math.sin(2 * math.pi * a / b))
        return np.array([[sample[0]], [sample[1]], [0]])

    def sampleFreeSpace(self):
        if random.randint(0, 100) > self.goalSampleRate:
            rnd = [random.uniform(self.minrand, self.maxrand),
                   random.uniform(self.minrand, self.maxrand)]
        else:
            rnd = [self.goal.x, self.goal.y]

        return rnd

    def getPathLen(self, path):
        pathLen = 0
        for i in range(1, len(path)):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen += math.sqrt((node1_x - node2_x)
                                 ** 2 + (node1_y - node2_y)**2)

        return pathLen

    def lineCost(self, node1, node2):
        return math.sqrt((node1.x - node2.x)**2 + (node1.y - node2.y)**2)

    def getNearestListIndex(self, nodes, rnd):
        dList = [(node.x - rnd[0])**2
                 + (node.y - rnd[1])**2 for node in nodes]
        minIndex = dList.index(min(dList))
        return minIndex

    def __CollisionCheck(self, newNode, obstacleList):
        for (ox, oy, size) in obstacleList:
            dx = ox - newNode.x
            dy = oy - newNode.y
            d = dx * dx + dy * dy
            if d <= size**2 + self.minDist**2:
                return False  # collision

        return True  # safe

    def getNewNode(self, theta, nind, nearestNode):
        newNode = copy.deepcopy(nearestNode)

        newNode.x += self.expandDis * math.cos(theta)
        newNode.y += self.expandDis * math.sin(theta)

        newNode.cost += self.expandDis
        newNode.parent = nind
        return newNode

    def isNearGoal(self, node):
        d = self.lineCost(node, self.goal)
        if d < self.expandDis:
            return True
        return False

    def rewire(self, newNode, nearInds):
        nnode = len(self.nodeList)
        for i in nearInds:
            nearNode = self.nodeList[i]

            d = math.sqrt((nearNode.x - newNode.x)**2
                          + (nearNode.y - newNode.y)**2)

            scost = newNode.cost + d

            if nearNode.cost > scost:
                theta = math.atan2(newNode.y - nearNode.y,
                                   newNode.x - nearNode.x)
                if self.check_collision_extend(nearNode, theta, d):
                    nearNode.parent = nnode - 1
                    nearNode.cost = scost

    def check_collision_extend(self, nearNode, theta, d):
        tmpNode = copy.deepcopy(nearNode)

        for i in range(int(d / self.expandDis)):
            tmpNode.x += self.expandDis * math.cos(theta)
            tmpNode.y += self.expandDis * math.sin(theta)
            if not self.__CollisionCheck(tmpNode, self.obstacleList):
                return False

        return True

    def getFinalCourse(self, lastIndex):
        path = [[self.goal.x, self.goal.y]]
        while self.nodeList[lastIndex].parent is not None:
            node = self.nodeList[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])
        return path

    def GenerateWaypointTime(self, path):
                
        waypointPathTime = []
        #print(path)
        for i in range(len(path)-1,0,-1):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen = math.sqrt((node1_x - node2_x)
                                 ** 2 + (node1_y - node2_y)**2)
            # Compute segment time based on the current self.accMax and pathLen
            segtime = math.sqrt((2*pathLen)/self.accMax)
            waypointPathTime.append(segtime)

        return waypointPathTime
        
    def GenerateWaypoint(self, path):
        pathLen = 0
        
        # Compute mini segment length based on the current self.accMax
        expand = (self.accMax*self.h**2)/2
        expand = self.minDist
        waypointPath = []
        for i in range(len(path)-1,0,-1):
            node1_x = path[i][0]
            node1_y = path[i][1]
            node2_x = path[i - 1][0]
            node2_y = path[i - 1][1]
            pathLen = math.sqrt((node1_x - node2_x)
                                 ** 2 + (node1_y - node2_y)**2)
            theta = math.atan2(node2_y-node1_y, node2_x-node1_x)
            waypointPath.append([node1_x,node1_y])
            temp_n = copy.deepcopy(path[i])

            while(pathLen > expand):
                temp_n[0] += expand * math.cos(theta)
                temp_n[1] += expand * math.sin(theta)
                waypointPath.append([temp_n[0],temp_n[1]])
                pathLen = pathLen - expand
            waypointPath.append([node2_x,node2_y])

        return waypointPath

    def OptimizeTraj(self, waypoints):
        # Initial precomputed path (waypoint based)
        wp = np.matrix(waypoints)

        # Number of states: position and velocity on x and y directions
        n = 4
        # Number of inputs: thrust on x and y directions
        m = 2

        # Number of iterations
        K = len(waypoints[0])
        print("Numar iteratii: %d" % K)
        
        # Total time
        T = self.h*K
        print("Timp de executie: %f" % T)

        Ldt = []
        for l in range(K+1):
            Ldt.append(l*self.h)
        #print(Ldt)

        # Kinematics of a point mass
        A = np.matrix([[1,0,self.h,0],[0,1,0,self.h],[0,0,1,0],[0,0,0,1]])

        B = np.matrix([[(self.h**2)/2,0],[0,(self.h**2)/2],[self.h,0],[0,self.h]])

        # Start state
        x_0 = np.array([self.start.x,self.start.y,0,0])
        x_f = np.array([self.goal.x,self.goal.y,0,0])

        # Form and solve control problem.
        x = cp.Variable((n, K+1))
        u = cp.Variable((m, K))

        cost = 0
        constr = []
        for t in range(0,K):
            cost += cp.sum_squares(u[:,t])

            constr += [x[:,t+1] == A@x[:,t] + B@u[:,t],
                       cp.norm(wp[:,t] - x[:2,t][:,None],2) <= self.minDist,
                       cp.norm(u[:,t], 2) <= self.accMax,
                       cp.norm(x[2:,t], 2) <= self.velMax]
        # sums problem objectives and concatenates constraints with the initial and final states.
        constr += [x[:,K] == x_f, x[:,0] == x_0]

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose = True)
        # Time taken until this point
        end_time = time.time()
        print("Exec time: %fs" % (end_time-start_time))

        # if problem.status not in ["infeasible", "unbounded"]:
        # # Otherwise, problem.value is inf or -inf, respectively.
        #     print("Optimal value: %s" % problem.value)
        # for variable in problem.variables():
        #     print("Variable %s: value %s" % (variable.name(), variable.value))

        plt.figure(2)
        # Plot (x_t)_3.
        plt.subplot(2,2,1)
        x3 = (x[2,:].value).tolist()
        #print(x3)
        plt.plot(Ldt,x3)
        #plt.axis([-self.velMax, T, -self.velMax, self.velMax])
        plt.ylabel(r"$v_x[m/s]$", fontsize=16)
        plt.ylim([-self.velMax, self.velMax])
        #plt.xticks([])
        plt.xlabel(r"$t[s]$", fontsize=16)
        plt.xlim([0, T])
        plt.grid(True)

        # Plot (x_t)_4.
        plt.subplot(2,2,2)
        x4 = x[3,:].value
        #print(x4)
        plt.plot(range(K+1), x4)
        plt.ylabel(r"$v_y[m/s]$", fontsize=16)
        plt.ylim([-self.velMax, self.velMax])
        #plt.xticks([])
        plt.xlabel(r"$t[s]$", fontsize=16)
        plt.xlim([0, T])
        
        plt.grid(True)

        # Plot (u_t)_1.
        plt.subplot(2,2,3)
        plt.plot(u[0,:].value)
        plt.ylabel(r"$a_x[m/(s*s)]$", fontsize=16)
        plt.ylim([-self.accMax, self.accMax])
        #plt.yticks(np.linspace(-1, 1, 3))
        #plt.xticks([])
        plt.xlabel(r"$t[s]$", fontsize=16)
        plt.xlim([0, T])
        plt.tight_layout()
        plt.grid(True)

        # Plot (u_t)_2.
        plt.subplot(2,2,4)
        plt.plot(u[1,:].value)
        plt.ylabel(r"$a_y[m/(s*s)]$", fontsize=16)
        plt.ylim([-self.accMax, self.accMax])
        #plt.yticks(np.linspace(-1, 1, 3))
        #plt.xticks([])
        plt.xlabel(r"$t[s]$", fontsize=16)
        plt.xlim([0, T])
        plt.tight_layout()
        plt.grid(True)

        stateMat = np.matrix([x[0,:].value,x[1,:].value,x[2,:].value,x[3,:].value])
        ctrlMat = np.matrix([u[0,:].value,u[1,:].value])

        #print(ctrlMat)
        #print(x)
        # Returnam matricea cu starile pentru a extrage din aceasta pozitia (x,y), in vederea reprezentarii grafice
        return stateMat

    def OptimizeTimeTraj(self, traj_time):

        # Number of states: position and velocity on x and y directions
        n = 4
        # Number of inputs: thrust on x and y directions
        m = 2

        # Number of iterations
        K = int(math.ceil(traj_time/self.h))
        print("Numar iteratii: %d" % K)
        
        # Total time
        print("Timp de executie: %f" % traj_time)

        Ldt = []
        for l in range(K+1):
            Ldt.append(l*self.h)
        #print(Ldt)

        # Kinematics of a point mass
        A = np.matrix([[1,0,self.h,0],[0,1,0,self.h],[0,0,1,0],[0,0,0,1]])

        B = np.matrix([[(self.h**2)/2,0],[0,(self.h**2)/2],[self.h,0],[0,self.h]])

        # Start state
        x_0 = np.array([self.start.x,self.start.y,0,0])

        # Form and solve control problem.
        x = cp.Variable((n, K+1))
        u = cp.Variable((m, K))

        cost = 0
        constr = []
        for t in range(0,K):
            cost += cp.sum_squares(u[:,t])

            constr += [x[:,t+1] == A@x[:,t] + B@u[:,t],
                       cp.norm(wp[:,t] - x[:2,t][:,None],2) <= self.minDist,
                       cp.norm(u[:,t], 2) <= self.accMax,
                       cp.norm(x[2:,t], 2) <= self.velMax]
        # sums problem objectives and concatenates constraints with the initial and final states.
        constr += [x[:,K] == np.array([self.goal.x,self.goal.y,0,0]), x[:,0] == x_0]

        problem = cp.Problem(cp.Minimize(cost), constr)
        problem.solve(verbose = True)
        # Time taken until this point
        end_time = time.time()
        print("Exec time: %fs" % (end_time-start_time))
        # if problem.status not in ["infeasible", "unbounded"]:
        # # Otherwise, problem.value is inf or -inf, respectively.
        #     print("Optimal value: %s" % problem.value)
        # for variable in problem.variables():
        #     print("Variable %s: value %s" % (variable.name(), variable.value))

        plt.figure(2)
        # Plot (x_t)_3.
        plt.subplot(2,2,1)
        x3 = (x[2,:].value).tolist()
        #print(x3)
        plt.plot(Ldt,x3)
        #plt.axis([-self.velMax, T, -self.velMax, self.velMax])
        plt.ylabel(r"$v_x[m/s]$", fontsize=16)
        plt.ylim([-self.velMax, self.velMax])
        #plt.xticks([])
        plt.xlabel(r"$t[s]$", fontsize=16)
        plt.xlim([0, traj_time])
        plt.grid(True)

        # Plot (x_t)_4.
        plt.subplot(2,2,2)
        x4 = x[3,:].value
        #print(x4)
        plt.plot(range(K+1), x4)
        plt.ylabel(r"$v_y[m/s]$", fontsize=16)
        plt.ylim([-self.velMax, self.velMax])
        #plt.xticks([])
        plt.xlabel(r"$t[s]$", fontsize=16)
        plt.xlim([0, traj_time])
        
        plt.grid(True)

        # Plot (u_t)_1.
        plt.subplot(2,2,3)
        plt.plot(u[0,:].value)
        plt.ylabel(r"$a_x[m/(s*s)]$", fontsize=16)
        plt.ylim([-self.accMax, self.accMax])
        #plt.yticks(np.linspace(-1, 1, 3))
        #plt.xticks([])
        plt.xlabel(r"$t[s]$", fontsize=16)
        plt.xlim([0, traj_time])
        plt.tight_layout()
        plt.grid(True)

        # Plot (u_t)_2.
        plt.subplot(2,2,4)
        plt.plot(u[1,:].value)
        plt.ylabel(r"$a_y[m/(s*s)]$", fontsize=16)
        plt.ylim([-self.accMax, self.accMax])
        #plt.yticks(np.linspace(-1, 1, 3))
        #plt.xticks([])
        plt.xlabel(r"$t[s]$", fontsize=16)
        plt.xlim([0, traj_time])
        plt.tight_layout()
        plt.grid(True)

        stateMat = np.matrix([x[0,:].value,x[1,:].value,x[2,:].value,x[3,:].value])
        ctrlMat = np.matrix([u[0,:].value,u[1,:].value])

        #print(ctrlMat)
        #print(x)
        # Returnam matricea cu starile pentru a extrage din aceasta pozitia (x,y), in vederea reprezentarii grafice
        return stateMat

    def drawGraph(self, xCenter=None, cBest=None, cMin=None, etheta=None, rnd=None):
        plt.figure(1)
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
            if cBest != float('inf'):
                self.plot_ellipse(xCenter, cBest, cMin, etheta)

        for node in self.nodeList:
            if node.parent is not None:
                if node.x or node.y is not None:
                    plt.plot([node.x, self.nodeList[node.parent].x], [
                        node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            #plt.plot(ox, oy, "oc", ms=30 * size)
            circle = plt.Circle((ox,oy),size/2, color='c')
            plt.gca().add_patch(circle)

        plt.plot(self.start.x, self.start.y, "^r",ms=5 * size)
        plt.plot(self.goal.x, self.goal.y, "^r",ms=5 * size)
        plt.axis('scaled')
        plt.axis([self.minrand, self.maxrand, self.minrand, self.maxrand])
        plt.xlabel(r"$x[m]$", fontsize=16)
        plt.ylabel(r"$y[m]$", fontsize=16)  
        plt.grid(True)
        #plt.pause(0.01)

    def plot_ellipse(self,xCenter, cBest, cMin, etheta):
        plt.figure(1)
        a = math.sqrt(cBest**2 - cMin**2) / 2.0
        b = cBest / 2.0
        angle = math.pi / 2.0 - etheta
        cx = xCenter[0]
        cy = xCenter[1]

        t = np.arange(0, 2 * math.pi + 0.1, 0.1)
        x = [a * math.cos(it) for it in t]
        y = [b * math.sin(it) for it in t]
        R = np.array([[math.cos(angle), math.sin(angle)],
                      [-math.sin(angle), math.cos(angle)]])
        fx = R @ np.array([x, y])
        px = np.array(fx[0, :] + cx).flatten()
        py = np.array(fx[1, :] + cy).flatten()
        plt.plot(cx, cy, "xm")
        plt.plot(px, py, ".m")

class Node():

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.cost = 0.0
        self.parent = None


def main():
    print("Start informed rrt star planning")

    # create obstacles
    # obstacleList = [
    #     (5, 5, 0.5),
    #     (9, 6, 1),
    #     (7, 5, 1),
    #     (2, 2, 1),
    #     (3, 6, 1),
    #     (7, 9, 1)
    # ]

    # obstacleList = [
    # (5, 5, 0),
    # (3, 6, 0),
    # (3, 8, 0),
    # (3, 10, 0),
    # (7, 5, 0),
    # (9, 5, 0)
    # ]  # [x,y,size(radius)]

    obstacleList = [
        (5, 5, 2),
        (3, 6, 2),
        (3, 8, 2),
        (3, 10, 2),
        (7, 5, 2),
        (9, 5, 2)
    ]  # [x,y,size(radius)]

    # Set params
    '''rrt = InformedRRTStar(start=[0, 0], goal=[3, 0],
                          randArea=[0, -5, 15], obstacleList=obstacleList)
    path = rrt.InformedRRTStarSearch(animation=show_animation)
    while path is None:
        path = rrt.InformedRRTStarSearch(animation=show_animation)
    waypointPath = rrt.GenerateWaypoint(path)
    #print([x for (x, y) in waypointPath], [y for (x, y) in waypointPath])
    waypointXaxis = [x for (x, y) in waypointPath]
    waypointYaxis = [y for (x, y) in waypointPath]

    waypointMatrix = [waypointXaxis,waypointYaxis]
    #print(waypointMatrix)
    optimizedPath = rrt.OptimizeTraj(waypointMatrix)'''
    
    rrt = InformedRRTStar(start=[0, 0], goal=[5, 10],
                          randArea=[-2, 15], obstacleList=obstacleList)
    path = rrt.InformedRRTStarSearch(animation=show_animation)
    while path is None:
        path = rrt.InformedRRTStarSearch(animation=show_animation)
    print(path)

    ''' Optimize traj based on current TIME trajectory '''
    #traj_time = rrt.GenerateWaypointTime(path)
    #optimizedPathTime = rrt.OptimizeTimeTraj(sum(traj_time))

    ''' Optimize traj based on current WP trajectory '''
    waypointPath = rrt.GenerateWaypoint(path)
    #print([x for (x, y) in waypointPath], [y for (x, y) in waypointPath])
    waypointXaxis = [x for (x, y) in waypointPath]
    waypointYaxis = [y for (x, y) in waypointPath]

    waypointMatrix = [waypointXaxis,waypointYaxis]
    #print(waypointMatrix)
    optimizedPath = rrt.OptimizeTraj(waypointMatrix)
    #print(optimizedPath[0,],optimizedPath[1,])
    #print(optimizedPath)
    #print(waypointPath)


    # Plot path
    if show_animation:
        plt.figure(1)
        #rrt.draw_graph()
        # Reprezentam grafic punctele determinate folosind metoda IRRT*
        plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        # Reprezentam grafic punctele GPS determinate, functie de constrangerile generate de scenariul de zbor
        plt.plot([x for (x, y) in waypointPath], [y for (x, y) in waypointPath], 'Hb')

        # Reprezentam grafic punctele GPS obtinute dupa aplicarea algoritmului de optimizare a traciectoriei
        plt.plot(optimizedPath[0,], optimizedPath[1,], '*y')
        #plt.plot(optimizedPathTime[0,], optimizedPathTime[1,], '*y')
        plt.xlabel(r"$x[m]$", fontsize=16)
        plt.ylabel(r"$y[m]$", fontsize=16)
        plt.grid(True)
        plt.show()

if __name__ == '__main__':
    start_time = time.time()
    main()
