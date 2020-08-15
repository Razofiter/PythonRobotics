"""
Path Planning Sample Code with Randamized Rapidly-Exploring Random Trees (RRT)

author: AtsushiSakai(@Atsushi_twi)

"""

import matplotlib.pyplot as plt
import random
import math
import copy
import time
from statistics import mean
from scipy.stats import norm

show_animation = True
start_time = 0.0


class RRT():
    """
    Class for RRT Planning
    """
    global start_time
    def __init__(self, start, goal, obstacleList,
                 randArea, expandDis=1.0, goalSampleRate=5, maxIter=200):
        """
        Setting Parameter

        start:Start Position [x,y]
        goal:Goal Position [x,y]
        obstacleList:obstacle Positions [[x,y,size],...]
        randArea:Ramdom Samping Area [min,max]

        """
        self.start = Node(start[0], start[1])
        self.end = Node(goal[0], goal[1])
        self.minrand = randArea[0]
        self.maxrand = randArea[1]
        self.expandDis = expandDis
        self.goalSampleRate = goalSampleRate
        self.maxIter = maxIter
        self.obstacleList = obstacleList

    def Planning(self, animation=True):
        """
        Pathplanning

        animation: flag for animation on or off
        """
        self.nodeList = [self.start]
        while True:
            # Random Sampling
            if random.randint(0, 100) > self.goalSampleRate:
                rnd = [random.uniform(self.minrand, self.maxrand), random.uniform(
                    self.minrand, self.maxrand)]
            else:
                rnd = [self.end.x, self.end.y]

            # Find nearest node
            nind = self.GetNearestListIndex(self.nodeList, rnd)
            # print(nind)

            # expand tree
            nearestNode = self.nodeList[nind]
            theta = math.atan2(rnd[1] - nearestNode.y, rnd[0] - nearestNode.x)

            newNode = copy.deepcopy(nearestNode)
            newNode.x += self.expandDis * math.cos(theta)
            newNode.y += self.expandDis * math.sin(theta)
            newNode.parent = nind

            if not self.__CollisionCheck(newNode, self.obstacleList):
                continue

            self.nodeList.append(newNode)
            # print("nNodelist:", len(self.nodeList))

            # check goal
            dx = newNode.x - self.end.x
            dy = newNode.y - self.end.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= self.expandDis:
                print("Goal!!")
                # print("--- %s seconds ---" % (time.time() - start_time))
                break

            # if animation:
            #     self.DrawGraph(rnd)

        path = [[self.end.x, self.end.y]]
        lastIndex = len(self.nodeList) - 1
        cost = 0.0
        timeRun = 0.0
        while self.nodeList[lastIndex].parent is not None:
            node = self.nodeList[lastIndex]
            path.append([node.x, node.y])
            lastIndex = node.parent
        path.append([self.start.x, self.start.y])

        # Compute path cost
        for k in range(len(path)-1):
            dx = path[k+1][0] - path[k][0]
            dy = path[k+1][1] - path[k][1]
            d = math.sqrt(dx ** 2 + dy ** 2)
            cost = cost + d
        
        timeRun = time.time() - start_time
        print("--- %s seconds ---" % timeRun)
        print (cost)
        return path,timeRun,cost

    def DrawGraph(self, rnd=None):
        """
        Draw Graph
        """
        plt.clf()
        if rnd is not None:
            plt.plot(rnd[0], rnd[1], "^k")
        for node in self.nodeList:
            if node.parent is not None:
                plt.plot([node.x, self.nodeList[node.parent].x], [
                         node.y, self.nodeList[node.parent].y], "-g")

        for (ox, oy, size) in self.obstacleList:
            plt.plot(ox, oy, "ok", ms=30 * size)

        plt.plot(self.start.x, self.start.y, "xr")
        plt.plot(self.end.x, self.end.y, "xr")
        plt.axis([-2, 15, -2, 15])
        plt.xlabel("x [m]")
        plt.ylabel("y [m]")
        plt.grid(True)
        plt.pause(0.01)

    def GetNearestListIndex(self, nodeList, rnd):
        dlist = [(node.x - rnd[0]) ** 2 + (node.y - rnd[1])
                 ** 2 for node in nodeList]
        minind = dlist.index(min(dlist))
        return minind

    def __CollisionCheck(self, node, obstacleList):

        for (ox, oy, size) in obstacleList:
            dx = ox - node.x
            dy = oy - node.y
            d = math.sqrt(dx * dx + dy * dy)
            if d <= size:
                return False  # collision

        return True  # safe


class Node():
    """
    RRT Node
    """

    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None


def main():
    global start_time
    print("start simple RRT path planning")

    # ====Search Path with RRT====
    # obstacleList = [
    #     (5, 5, 1),
    #     (3, 6, 2),
    #     (3, 8, 2),
    #     (3, 10, 2),
    #     (7, 5, 2),
    #     (9, 5, 2)
    # ]  # [x,y,size(radius)]

    obstacleList = [
        (5, 5, 2),
        (3, 10, 2),
        (9, 5, 1)
    ]  # [x,y,size]
    # Set Initial parameters
    rrt = RRT(start=[0, 0], goal=[5, 10],
              randArea=[-2, 15], obstacleList=obstacleList)
    timeRunList = []
    costList = []
    # Run the algorithm 50 times
    for n in range(10000):
        path , timeRun, cost = rrt.Planning(animation=show_animation)
        timeRunList.append(timeRun)
        costList.append(cost)
        start_time = time.time()

        # Draw final path
        # if show_animation:
        #     rrt.DrawGraph()
        #     plt.plot([x for (x, y) in path], [y for (x, y) in path], '-r')
        #     plt.grid(True)
            # plt.show()
    print("Cost list >>>" + str(costList))
    print("Time list >>>" + str(timeRunList))
    print("Min cost >>>" + str(min(costList)))
    print("Max cost >>>" + str(max(costList)))
    print("Average cost >>>" + str(mean(costList)))
    print("Min time >>>" + str(min(timeRunList)))
    print("Max time >>>" + str(max(timeRunList)))
    print("Average time >>>" + str(mean(timeRunList)))
    mu, std = norm.fit(costList)
    print("Mean value>>" + str(mu))
    print("Standard deviation >>" + str(std))
    plt.figure(1)
    plt.hist(costList, bins=int(mu))
    plt.xlabel("Valoarea de cost [m]")
    plt.ylabel("Numar aparitii")
    plt.grid()
    # plt.show()
    
    plt.figure(2)
    plt.plot(costList, norm.pdf(costList,mu,std))
    plt.xlabel("Valoarea de cost [m]")
    plt.ylabel("Densitatea functiei de distibutie")
    plt.grid()
    plt.show()
    # data_normal = norm.rvs(size=10000,loc=0,scale=1)
    # print(data_normal)
    # ax = sns.distplot(data_normal,
    #               bins=100,
    #               kde=True,
    #               color='skyblue',
    #               hist_kws={"linewidth": 15,'alpha':1})
    # ax.set(xlabel='Normal Distribution', ylabel='Frequency')


if __name__ == '__main__':
    start_time = time.time()
    main()
