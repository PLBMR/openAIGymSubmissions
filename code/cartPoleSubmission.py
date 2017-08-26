#cartPoleSubmission.py
#contains my code for performing CartPole-v0

#imports

import numpy as np
import gym
import sys

#helpers

#linear basis functions

def linearBasisFunc(stateVec,action):
    #helper for generating linear basis calculation
    #action is integer, stateVec is a vector
    stateMulVec = np.zeros(stateVec.shape[0] * 2)
    for i in range(stateVec.shape[0]):
        stateMulVec[(i * 2)] = stateVec[i] * action
        stateMulVec[(i * 2) + 1] = stateVec[i] * (1 - action)
    return stateMulVec

#classes

class LinearQFunc:
    #helper that holds our q function, a linear approximator
    def __init__(self,basisFunc):
        #helper that prepares our q function
        self.basisFunc = basisFunc

    def q(self,stateVec,action,weightVec):
        return np.dot(weightVec,self.basisFunc(stateVec,action))

    def gradQ(self,stateVec,action,weightVec):
        #since linear, it is just the basis function
        return self.basisFunc(stateVec,action)

class Agent
#main process

if __name__ == "__main__":
    #apiKey = sys.argv[1]
    #numEpisodes = sys.argv[2]
    #tests
    stateVec = np.array([1,2,3,4])
    action = 1
    print linearBasisFunc(stateVec,action)
    testQ = LinearQFunc(linearBasisFunc)
    weightVec = np.array([0,0,0,0,1,1,1,1])
    print testQ.q(stateVec,action,weightVec)
    print testQ.gradQ(stateVec,action,weightVec)
