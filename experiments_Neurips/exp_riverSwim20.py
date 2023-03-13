from learners.UCRL_RM import *
from learners.UCRL_CP import *
from learners.PSRL_RM import *
from learners.Optimal import *
#from learners.ImprovedMDPLearner2 import *
from utils import *

def run_exp(rendermode='', testName = "riverSwim6_patrol2", sup = ''):
	timeHorizon=100000
	nbReplicates=10
	
	if testName == "2-room_patrol2":
		env, nbS, nbA = buildGridworld_RM(sizeX=9,sizeY=11,map_name="2-room_patrol2",rewardStd=0.01, initialSingleStateDistribution=True)
	elif testName == "riverSwim6_patrol2":
		env, nbS, nbA = buildRiverSwim_patrol2(nbStates=4, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
	elif testName == "riverSwim20_patrol2":
		env, nbS, nbA = buildRiverSwim_patrol2(nbStates=20, rightProbaright=0.4, rightProbaLeft=0.05, rewardL=0.005, rewardR=1.)
	elif testName == "flower8":
		env, nbS, nbA = buildFlower(sizeB = 8, delta = 0.2)
	

	#if testName == "2-room_patrol2" or testName == "4-room":
	#	equivalence.displayGridworldEquivalenceClasses(env.env, 0.)
	#	equivalence.displayGridworldAggregationClasses(env.env)
	#C, nC = equivalence.compute_C_nC(env.env)
	
	#profile_mapping = equivalence.compute_sigma(env.env)
	#sizeSupport = equivalence.compute_sizeSupport(env.env)
	#print(env.env.maze)
	
	print("sup = ", sup)
	
	print("*********************************************")
	
	cumRewards = []
	names = []
	
	#learner1 = TSDE_CP( nbS,nbA, env.rewardMachine, delta=0.05)#, c = 10)
	#names.append(learner1.name())
	#cumRewards1 = cumulativeRewards_RM(env,learner1,nbReplicates,timeHorizon,rendermode)#, reverse = True)
	#cumRewards.append(cumRewards1)
	#pickle.dump(cumRewards1, open(("results/cumRewards_" + testName + "_" + learner1.name() + "_" + str(timeHorizon) + sup), 'wb'))
	#cumRewards1 = pickle.load(open(("results/cumRewards_" + testName + "_" + learner1.name() + "_" + str(timeHorizon) + sup), 'rb'))
	#cumRewards.append(cumRewards1)

	learner2 = TSDE_CP_unknown( nbS,nbA, env.rewardMachine, delta=0.05)
	names.append("TSDE(CP)")#learner2.name())
	cumRewards2 = cumulativeRewards(env,learner2,nbReplicates,timeHorizon,rendermode)#, reverse = True)
	cumRewards.append(cumRewards2)
	#pickle.dump(cumRewards2, open(("results/cumRewards_" + testName + "_" + learner2.name() + "_" + str(timeHorizon) + sup), 'wb'))
	
	learner3 = UCRL2_CP( nbS,nbA, env.rewardMachine, delta=0.05)
	names.append("UCRL2(CP)")#learner3.name())
	cumRewards3 = cumulativeRewards(env,learner3,nbReplicates,timeHorizon,rendermode)#, reverse = True)
	cumRewards.append(cumRewards3)
	#pickle.dump(cumRewards3, open(("results/cumRewards_" + testName + "_" + learner3.name() + "_" + str(timeHorizon) + sup), 'wb'))
	
	learner4 = UCRL2_RM( nbS,nbA, env.rewardMachine, delta=0.05)#, c = 10)
	names.append("UCRL2-RM-L")#learner4.name())
	cumRewards4 = cumulativeRewards(env,learner4,nbReplicates,timeHorizon,rendermode)#, reverse = True)
	cumRewards.append(cumRewards4)
	#pickle.dump(cumRewards4, open(("results/cumRewards_" + testName + "_" + learner4.name() + "_" + str(timeHorizon) + sup), 'wb'))
	
	#learner5 = UCRL2_RM_Bernstein( nbS,nbA, env.rewardMachine, delta=0.05)#, c = 10)
	#£names.append("UCRL2-RM-B")#learner5.name())
	#cumRewards5 = cumulativeRewards_RM(env,learner5,nbReplicates,timeHorizon,rendermode)#, reverse = True)
	#cumRewards.append(cumRewards5)
	#pickle.dump(cumRewards5, open(("results/cumRewards_" + testName + "_" + learner5.name() + "_" + str(timeHorizon) + sup), 'wb'))

	#learner6 = UCRL2_CP_Bernstein( nbS,nbA, env.rewardMachine, delta=0.05)
	#names.append("UCRL2-B(CP)")#learner6.name())
	#cumRewards6 = cumulativeRewards(env,learner6,nbReplicates,timeHorizon,rendermode)#, reverse = True)
	#cumRewards.append(cumRewards6)
	#pickle.dump(cumRewards6, open(("results/cumRewards_" + testName + "_" + learner6.name() + "_" + str(timeHorizon) + sup), 'wb'))
	
	#learner7 = TSDE_RM( nbS,nbA, env.rewardMachine, delta=0.05)
	#names.append(learner7.name())
	#cumRewards7 = cumulativeRewards(env,learner7,nbReplicates,timeHorizon,rendermode)
	#cumRewards.append(cumRewards7)
	#pickle.dump(cumRewards7, open(("results/cumRewards_" + testName + "_" + learner7.name() + "_" + str(timeHorizon) + sup), 'wb'))


	if testName == "2-room_patrol2":
		opti_learner = Opti_911_2room_patrol2(env.env)
	elif testName == "riverSwim6_patrol2" or testName == "riverSwim20_patrol2":
		opti_learner = Opti_riverSwim_patrol2(env.env)
	elif testName == "random_grid" or testName == "2-room" or testName == "4-room":
		print("Computing an estimate of the optimal policy (for regret)...")
		opti_learner = Opti_learner(env.env, nbS, nbA)
		print("Done, the estimation of the optimal policy : ")
		print(opti_learner.policy)
	elif testName == "flower8":
		opti_learner = Opti_flower(env.env)
	else:
		opti_learner = Opti_swimmer(env)
	
	cumReward_opti = cumulativeRewards_RM(env,opti_learner,1,min((1000000, 5 * timeHorizon)),rendermode)
	gain =  cumReward_opti[0][-1] / (min((1000000, 5 * timeHorizon)))#compute_gstar_samdp(env)
	print(gain)
	opti_reward = [[t * gain for t in range(timeHorizon)]]
	
	cumRewards.append(opti_reward)
	print('About to plot')
	plotCumulativeRegrets(names, cumRewards, timeHorizon, testName)
	plotCumulativeRegrets(names, cumRewards, timeHorizon, testName, ysemilog=True)
	
	
	print("*********************************************")

#run_exp(rendermode='pylab')	#Pylab rendering
#run_exp(rendermode='text')	#Text rendering
run_exp(rendermode='', testName ='riverSwim20_patrol2', sup = '_0')