#cartPoleSubmission.py
#contains my code for performing CartPole-v0

#imports

import numpy as np
import gym
import sys
import random
import scipy.misc as spm

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

def firstInteractionBasisFunc(stateVec,action):
    #helper for generating a linear basis along with a basis of interactions
    numStates = stateVec.shape[0]
    numLinearObs = spm.comb(numStates,1) * 2
    numInterObs = spm.comb(numStates,2) * 2
    stateMulVec = np.zeros((numLinearObs + numInterObs).astype("int"))
    intIndexLookup = {(0,1):0,(0,2):2,(0,3):4,(1,2):6,(1,3):8,(2,3):10}
    #first get linear component
    for i in range(numStates):
        stateMulVec[(i * 2)] = stateVec[i] * action
        stateMulVec[(i * 2) + 1] = stateVec[i] * (1 - action)
    #then get interaction component
    for i in range(numStates):
        for j in range(i+1,numStates):
            stateLookup = (numLinearObs + intIndexLookup[(i,j)]).astype("int")
            stateMulVec[stateLookup] = stateVec[i] * stateVec[j] * action
            stateMulVec[stateLookup + 1] = (stateVec[i]*stateVec[j]*(1-action))
    return stateMulVec

def firstIntWithPolynomials(stateVec,action):
    #helper for generating a linear basis along with a basis of interactions and
    #second degree polynomials
    numStates = stateVec.shape[0]
    numLinearObs = numPolyObs = spm.comb(numStates,1) * 2
    numInterObs = spm.comb(numStates,2) * 2
    stateMulVec = np.zeros((numLinearObs + numInterObs + numPolyObs).astype(
                                                                        "int"))
    intIndexLookup = {(0,1):0,(0,2):2,(0,3):4,(1,2):6,(1,3):8,(2,3):10}
    #first get linear component
    for i in range(numStates):
        stateMulVec[(i * 2)] = stateVec[i] * action
        stateMulVec[(i * 2) + 1] = stateVec[i] * (1 - action)
    #then get interaction component
    for i in range(numStates):
        for j in range(i+1,numStates):
            stateLookup = (numLinearObs + intIndexLookup[(i,j)]).astype("int")
            stateMulVec[stateLookup] = stateVec[i] * stateVec[j] * action
            stateMulVec[stateLookup + 1] = (stateVec[i]*stateVec[j]*(1-action))
    #then get first degree polynomials
    stateStart = (numLinearObs + numInterObs).astype("int")
    for i in range(numStates):
        stateMulVec[stateStart+(i * 2)] = (stateVec[i]**2) * action
        stateMulVec[stateStart+(i * 2) + 1] = (stateVec[i]**2) * (1 - action)
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
        #choose initial weights
        if basisFunc == linearBasisFunc:
            initWeightVec = np.zeros(len(self.env.observation_space.high) * 2)
        elif basisFunc == firstInteractionBasisFunc: #interaction one
            observationSpaceSize = len(self.env.observation_space.high)
            numLinearTerms = observationSpaceSize * 2
            numInteractionTerms = spm.comb(observationSpaceSize,2) * 2
            numTerms = (numLinearTerms + numInteractionTerms).astype("int")
            initWeightVec = np.zeros(numTerms)
        else: #with polynomials
            observationSpaceSize = len(self.env.observation_space.high)
            numLinearTerms = numPolyTerms = observationSpaceSize * 2
            numInteractionTerms = spm.comb(observationSpaceSize,2) * 2
            numTerms = (numLinearTerms + numInteractionTerms 
                        + numPolyTerms).astype("int")
            initWeightVec = np.zeros(numTerms)
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
    #stateVec = np.array([1,2,3,4])
    #action = 1
    #print firstIntWithPolynomials(stateVec,action)
    #testQ = LinearQFunc(linearBasisFunc)
    #weightVec = np.array([0,0,0,0,1,1,1,1])
    #print testQ.q(stateVec,action,weightVec)
    #print testQ.gradQ(stateVec,action,weightVec)
    epsilon = .001
    alpha = .5
    gamma = 1
    newInteraction = AgentEnvironmentInteraction("CartPole-v0",alpha,gamma,
                                                 epsilon,
                                                 firstInteractionBasisFunc,
                                                 "../submission/cp-e-16")
    newInteraction.performMultipleEpisodes(800)
    newInteraction.env.close()
    gym.upload("../submission/cp-e-16",api_key = sys.argv[1])
