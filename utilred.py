# -*- coding: utf-8 -*-
"""
Created on Thu Jan 21 11:38:05 2021

@author: Philip Brown
"""

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import patches
import numpy as np
import numpy.random as rnd

class Agent :
    def __init__(self,index,blindspots) :
        self.index = index
        self.blindspots = blindspots
        self.location = [0,0]

class Game :
    def __init__(self,n,k,points,radius,agentRadius,deny=False,seed=0) :
        # n: number of agents
        # k: number of peaks in objective function
        # points: number of discrete points in each dimension of action space
        self.action_history = []
        self.W_history = []
        self.n = n
        self.k = k
        self.points = points
        self.agents = [Agent(i,set()) for i in range(n)]
        self.agentRadius = agentRadius
        self.objective_arr = np.zeros([points,points])
        self.peaks = []
        rnd.seed(seed)
        for peak in range(k) :
            center = self.random_location()
            center_x = center[0]
            center_y = center[1]
            self.peaks.append(center)
            for x in range(center_x-radius,center_x+radius) :
                for y in range(center_y-radius,center_y+radius) :
                    try :
                        self.objective_arr[x,y] += (1-abs(x-center_x)/radius)*(1-abs(y-center_y)/radius)
                    except IndexError :
                        pass
    
    def random_location(self) :
        x = rnd.randint(0,self.points)
        y = rnd.randint(0,self.points)
        return (x,y)
    
    def addx(self,loc,delta) :
        # adds delta to 1st component of loc. Cannot return out of bounds.
        return (min(self.points-1,max(0,loc[0]+delta)),loc[1])
    
    def addy(self,loc,delta) :
        # adds delta to 2st component of loc. Cannot return out of bounds.
        return (loc[0],min(self.points-1,max(0,loc[1]+delta)))
    
    def objective(self,loc) :
        # exception handling ensures out-of-range indices give 0 objective
        try :
            return self.objective_arr[loc[0],loc[1]]
        except IndexError :
            return 0
    
    def W(self,ignoreAgents=set()) :
        # takes the union of all points covered by agents, then adds their obj value
        # ignores agents in set ignoreAgents -- use for marginal contribution and commfail
        covered = set()
        Wval = 0
        for i,agent in enumerate(self.agents) :
            if i not in ignoreAgents :
                for x in range(agent.location[0]-self.agentRadius,agent.location[0]+self.agentRadius) :
                    for y in range(agent.location[1]-self.agentRadius,agent.location[1]+self.agentRadius) :
                        covered.add((x,y))
        for loc in covered :
            Wval += self.objective(loc)
        return Wval
    
    def overlapping(self,loc1,loc2) :
        # return True if agents at loc1 and loc2 overlap each other
        if abs(loc1[0]-loc2[0]) <= self.agentRadius*2 :
            if abs(loc1[1]-loc2[1]) <= self.agentRadius*2 :
                return True
        return False

    def get_irrelevant_agents(self,i,loc) :
        # add logic for commfails!
        # returns a set of agents that do NOT overlap agent i at location loc
        irrelevant_agents = set()
        for j,agent in enumerate(self.agents) :
            if j!=i :
                if not self.overlapping(loc,agent.location) :
                    irrelevant_agents.add(j)
        return irrelevant_agents
    
    def utility(self,i,loc) :
        # i is index of agent
        # crude: it literally moves the agent to new location, computes MC, then moves back
        # write first for simple MC
        # also, it computes MC inefficiently: modify to simply look at area covered by agent
        current_location = self.agents[i].location
        self.agents[i].location = loc
        # irrelevant = set() #
        irrelevant = self.get_irrelevant_agents(i,loc)
        # print(i)
        # print(irrelevant)
        U = self.W(irrelevant) - self.W(irrelevant.union({i})) # marginal contribution
        self.agents[i].location = current_location
        return U
    
    def best_response(self,i) :
        best_seen = (0,0)
        best_utility = self.utility(i,(0,0))
        for x in range(1,self.points) :
            for y in range(0,self.points) :
                util = self.utility(i,(x,y))
                if util > best_utility :
                    best_seen = (x,y)
                    best_utility = util
        return best_seen
    
    def best_response_step(self,i) :
        # moves agent to an arbitrary location in its best response set.
        # If agent moves, return True.
        currentLoc = self.agents[i].location
        best = self.best_response(i)
        if best == currentLoc :
            return False
        else :
            self.agents[i].location = best
            return True
        
    def best_response_run(self,numSteps=2) :
        # numSteps is max iterations thru agent list
        # returns True if converged to Nash NASH LOGIC CURRENTLY BROKEN
        # returns False if not Nash
        for t in range(numSteps) :
            Nash = True  # maybe we're at Nash?
            for i,agent in enumerate(self.agents) :
                print('agent '+str(i))
                moved = self.best_response_step(i)
                Nash = Nash and not moved # if someone moves, we weren't at Nash
                self.W_history.append(self.W())
            if Nash:
                return Nash # still true if everybody didn't move
        return Nash
    
    def better_reply_step(self,i,loc) :
        # checks utility at new location; if an improvement, move
        # returns True if moved, False if not moved
        currentU = self.utility(i,self.agents[i].location)
        candidateU = self.utility(i,loc) 
        if candidateU > currentU :
            self.agents[i].location = loc
            return True
        return False
    
            
    def better_reply_run(self,numSteps=1000,localsearch=False) :
        # one step cycles through all agents
        # localsearch=True causes agent to check 4 compass points for a better reply, then check random location
        for t in range(numSteps) :
            for i,agent in enumerate(self.agents) :
                if localsearch :
                    current_location = agent.location
                    compass_points = [self.addx(current_location,1),
                                      self.addx(current_location,-1),
                                      self.addy(current_location,1),
                                      self.addy(current_location,-1)]
                    for point in compass_points :
                        if self.better_reply_step(i,point) :
                            break # stop local search if we moved
                loc = self.random_location()
                self.better_reply_step(i,loc)
                self.W_history.append(self.W())
        
    
    def plotObjective3d(self,fignum=13) :
        x = np.arange(0,self.points)
        y = x.copy()
        X,Y = np.meshgrid(x,y)
        
        fig = plt.figure(fignum)
        ax = fig.gca(projection='3d')
        ax.plot_surface(X,Y,self.objective_arr,cmap=cm.coolwarm)
    
    def plotObjective2d(self,agents=False,annot=False,fignum=12) :
        # if agents=True, then each agent is represented by a box showing it sensing radius
        # if annot=True, then the agent index is printed on the box
        x = np.arange(0,self.points)
        y = x.copy()
        X,Y = np.meshgrid(x,y)
        plt.figure(fignum)
        plt.pcolormesh(X,Y,self.objective_arr,shading='auto')
        
        if agents:
            ax = plt.gca()
            for agent in self.agents :
                xy = (max(0,agent.location[1] - np.floor(self.agentRadius)),
                      max(0,agent.location[0] - np.floor(self.agentRadius)))
                rect = patches.Rectangle(xy, self.agentRadius*2, self.agentRadius*2,
                                         edgecolor='white',facecolor='none',lw=1)
                ax.add_patch(rect)
                if annot:
                    xy = (agent.location[1],agent.location[0])
                    ax.annotate(str(agent.index),xy=xy)
    
    def plotWhist(self,fignum=17) :
        plt.figure(17)
        plt.plot(self.W_history)
        
        
if __name__ == "__main__" :
    numSteps = 100
    game = Game(5,3,100,40,4,seed=0)
    game.plotObjective2d()
    game.best_response_run(numSteps)
    game.plotObjective2d(True,True)
    game.plotWhist()
        
# bigger game:
# if __name__ == "__main__" :
#     numSteps = 100
#     game = Game(10,5,200,80,8,seed=2)
#     game.plotObjective2d()
#     game.better_reply_run(numSteps,True)
#     game.plotObjective2d(True,True)
#     game.plotWhist()