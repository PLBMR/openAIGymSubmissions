#cartPoleSubmission.py
#contains my code for performing CartPole-v0

#imports

import numpy as np
import gym
import sys
import random

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

class Agent:
    #defines the interactor with the environment
    def __init__(self,actionSet,weightVec,alpha,gamma,epsilon,basisFunc):
        self.actionSet = actionSet
        self.weightVec = weightVec
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.qFunc = LinearQFunc(basisFunc)
        #places to store states and actions
        self.nextState = None
        self.nextAction = None
        self.reward = 0
        self.prevState = None
        self.prevAction = None

    def chooseAction(self,state): #helper that makes an action decision based on
        #a given state
        randomNum = random.uniform(0,1)
        if (randomNum > 1 - self.epsilon): #choose random policy
            randomAction = random.sample(self.actionSet,1)[0]
            return randomAction
        else: #choose greedy policy
            greedyAction = None
            greedyActionVal = None
            for action in self.actionSet:
                tempActionVal = self.qFunc.q(state,action,self.weightVec)
                if (type(greedyActionVal) == type(None)
                    or greedyActionVal < tempActionVal): #found better action
                    greedyActionVal = tempActionVal
                    greedyAction = action
            return greedyAction

    def takeAction(self,nextState,nextReward,done): #hepler that makes an action
        #decision and an update after observing R and S'
        self.nextState = nextState
        self.reward = nextReward
        if (done): #just need to do weight update
            qVal = self.qFunc.q(self.prevState,self.prevAction,self.weightVec)
            qGradVec = self.qFunc.gradQ(self.prevState,self.prevAction,
                                        self.weightVec)
            self.weightVec = (self.weightVec + self.alpha * (self.reward
                                                            - qVal)
                              * qGradVec)
            return None #no action taken
        else: #need to choose next action and weight update
            self.nextAction = self.chooseAction(self.nextState)
            nextQVal = self.qFunc.q(self.nextState,self.nextAction,
                                    self.weightVec)
            prevQVal = self.qFunc.q(self.prevState,self.prevAction,
                                    self.weightVec)
            prevQVec = self.qFunc.gradQ(self.prevState,self.prevAction,
                                        self.weightVec)
            self.weightVec = (self.weightVec 
                + self.alpha * (self.reward + self.gamma * nextQVal - prevQVal) 
                * prevQVec)
            #store our next and previous action
            self.prevState = self.nextState
            self.prevAction = self.nextAction
            return self.nextAction

class AgentEnvironmentInteraction:
    def __init__(self,gameName,alpha,gamma,epsilon,basisFunc,
                 monitorFilename = None):
        #helper for initializing our environment
        self.env = gym.make(gameName)
        if (type(monitorFilename) != type(None)):
            self.env = gym.wrappers.Monitor(self.env,monitorFilename)
        initWeightVec = np.zeros(len(self.env.observation_space.high) * 2)
        actionSet = set(range(self.env.action_space.n))
        self.agent = Agent(actionSet,initWeightVec,alpha,gamma,epsilon,basisFunc
                          )
        #then some meta information
        self.episodeLengthVec = []
        self.episodeRewardVec = []

    def performEpisode(self): #helper for performing a given episode
        #start meta-parameters
        episodeLength = 0
        totalReward = 0
        done = False #will alter this
        #start initial state and action
        initState = self.env.reset()
        self.agent.prevState = initState
        self.agent.prevAction = self.agent.chooseAction(self.agent.prevState)
        while (not(done)): #run a step
            episodeLength += 1
            #self.env.render()
            nextState, nextReward, done, _ = self.env.step(
                                                        self.agent.prevAction)
            totalReward += nextReward
            nextAction = self.agent.takeAction(nextState,nextReward,done)
        #then update meta-parameters
        self.episodeLengthVec.append(episodeLength)
        self.episodeRewardVec.append(totalReward)

    def performMultipleEpisodes(self,numEpisodes): #helper for performing
        #multiple episodes
        for episode in range(numEpisodes):
            self.performEpisode()
        
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
    epsilon = .1
    alpha = 1
    gamma = .9
    newInteraction = AgentEnvironmentInteraction("CartPole-v0",alpha,gamma,
                                                 epsilon,linearBasisFunc,
                                                 "../submission/cp-e-5")
    newInteraction.performMultipleEpisodes(20000)
    newInteraction.env.close()
    gym.upload("../submission/cp-e-5",api_key = sys.argv[1])
