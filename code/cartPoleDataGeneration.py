#cartPoleDataGeneration.py
#helper for performing a grid search over model parameters for the cartpole
#submission

import cartPoleSubmission as cps
import pickle as pkl

if __name__ == "__main__":
    #get vectors of models considered
    epsilonVec = [.4,.2,.1,.05,.01]
    alphaVec = [.5,.6,.7,.8,.9,1]
    gammaVec = [.3,.4,.5,.6,.7,.8,.9,1]
    #initialize model dictionary
    modelDict = {}
    episodeLengthDict = {}
    rewardDict = {}
    index = 0 #will add to this
    numEpisodes = 400
    for e in epsilonVec:
        for a in alphaVec:
            for g in gammaVec:
                numRuns = 1
                for run in range(numRuns):
                    modelDict[index] = {"epsilon":e,"alpha":a,"gamma":g}
                    #perform run
                    newInteraction = cps.AgentEnvironmentInteraction(
                                                "CartPole-v0",
                                            a,g,e,cps.firstInteractionBasisFunc)
                    newInteraction.performMultipleEpisodes(numEpisodes)
                    #then extract results
                    episodeLengthVec = newInteraction.episodeLengthVec
                    rewardVec = newInteraction.episodeRewardVec
                    #then store
                    episodeLengthDict[index] = episodeLengthVec
                    rewardDict[index] = rewardVec
                    index += 1
    #then export
    pkl.dump(modelDict,open("../data/raw/simpleModelDict.pkl","wb"))
    pkl.dump(episodeLengthDict,open("../data/raw/simpleEpisodeLengthDict.pkl",
                                    "wb"))
    pkl.dump(rewardDict,open("../data/raw/simpleRewardDict.pkl","wb"))
