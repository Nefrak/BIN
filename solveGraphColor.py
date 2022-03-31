from deap import base
from deap import creator
from deap import tools

import random
import numpy

import matplotlib.pyplot as plt
import seaborn as sns

import elitism
import graphs

# problem constants:
HARD_CONSTRAINT_PENALTY = 10  # the penalty factor for a hard-constraint violation
FILE_PATH = "params.yaml"
SHOW_GRAPH = False
SHOW_CONV_STAT = False
SHOW_BOX_STAT = False

# Genetic Algorithm constants:
POPULATION_SIZE = 100
P_CROSSOVER = 0.9  # probability for crossover
P_MUTATION = 0.1   # probability for mutating an individual
RUNS = 1
MAX_GENERATIONS = 100
HALL_OF_FAME_SIZE = 5
MAX_COLORS = 9

# set the random seed:
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

toolbox = base.Toolbox()

# create the graph coloring problem instance to be used:
gcp = graphs.GraphColoringProblem(FILE_PATH, HARD_CONSTRAINT_PENALTY)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)

# create an operator that randomly returns an integer in the range of participating colors:
toolbox.register("Integers", random.randint, 0, MAX_COLORS - 1)

# create the individual operator to fill up an Individual instance:
toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Integers, len(gcp))

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# fitness calculation: cost of the suggested olution
def getCost(individual):
    return gcp.getCost(individual),  # return a tuple

#custom genetic functions
def mutateConflictNodes(individual, low, up, indpb):
    for index in range(len(individual)):
        if gcp.isInViolation(index, individual) != -1 and random.random() < indpb:
            individual[index] = random.randrange(low, up + 1)
    return individual

def mutateSwitchConflictNodes(individual, indpb):
    for index in range(len(individual)):
        confIndex = gcp.isInViolation(index, individual)
        if confIndex != -1 and random.random() < indpb:
            savedColor = individual[index]
            individual[index] = individual[confIndex]
            individual[confIndex] = savedColor
    return individual

def crossNoConflict(ind1, ind2):
    for index in range(len(ind1)):
        if gcp.isInViolation(index, ind1) != -1:
            ind1[index] = ind2[index]
    for index in range(len(ind2)):
        if gcp.isInViolation(index, ind2) != -1:
            ind2[index] = ind1[index]
    return (ind1, ind2)

toolbox.register("evaluate", getCost)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
#toolbox.register("select", tools.selBest)

#toolbox.register("mate", tools.cxOnePoint)
toolbox.register("mate", tools.cxTwoPoint)
#toolbox.register("mate", crossNoConflict)

#toolbox.register("mutate", tools.mutUniformInt, low=0, up=MAX_COLORS - 1, indpb=1.0/len(gcp))
#toolbox.register("mutate", mutateConflictNodes, low=0, up=MAX_COLORS - 1, indpb=1.0/len(gcp))
toolbox.register("mutate", mutateSwitchConflictNodes, indpb=1.0/len(gcp))

def showConv(logbooks, runs):
    # extract statistics:
    minAvg = numpy.array(logbooks[0].select("min"))
    meanAvg = numpy.array(logbooks[0].select("avg"))
    maxAvg = numpy.array(logbooks[0].select("max"))
    for i in range(1, runs):
        minAvg = minAvg + numpy.array(logbooks[i].select("min"))
        meanAvg = meanAvg + numpy.array(logbooks[i].select("avg"))
        maxAvg = maxAvg + numpy.array(logbooks[i].select("max"))
    minFitnessValues = minAvg / runs
    meanFitnessValues = meanAvg / runs
    maxFitnessValues = maxAvg / runs

    plt.figure(1)
    sns.set_style("whitegrid")
    x = numpy.arange(0, MAX_GENERATIONS + 1)
    y = [minFitnessValues, meanFitnessValues, maxFitnessValues]
    plt.xscale('log')
    plt.plot(x,y[1],color='C0')
    plt.fill_between(x,y[0],y[2], color='C0',alpha=0.4)
    plt.axhline(minFitnessValues[-1], color="black", linestyle="--")
    plt.xlabel('Pocet evaluaci')
    plt.ylabel('Fitness')
    plt.title('Konvergencni krivka')

def showBox(logbooks, runs):
    lastEval = [logbooks[0].select("avg")[-1]]
    for i in range(1, runs):
        lastEval.append(logbooks[i].select("avg")[-1])

    plt.figure(2)
    x1 = lastEval
    plt.boxplot([x1], labels=['x1'], notch=True)
    plt.xlabel('Fitness')
    plt.ylabel('Outputs')
    plt.title('Boxplot')

def printResult(hofs, logbooks, runs):
    # print info for best solution found:
    best = hofs[0].items[0]
    for hof in hofs:
        if best.fitness.values[0] < hof.items[0].fitness.values[0]:
            best = hof.items[0]

    print("-- Best Individual = ", best)
    print("-- Best Fitness = ", best.fitness.values[0])
    print("number of colors = ", gcp.getNumberOfColors(best))
    print("Number of violations = ", gcp.getViolationsCount(best))
    print("Cost = ", gcp.getCost(best))

    # plot statistics:
    if SHOW_CONV_STAT:
        showConv(logbooks, runs)
             
    if SHOW_BOX_STAT:
        showBox(logbooks, runs)

    # plot best solution:
    if SHOW_GRAPH:
        plt.figure(3)
        gcp.plotGraph(best)

    plt.show()

# Genetic Algorithm flow:
def main():
    # prepare the statistics object:
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("min", numpy.min)
    stats.register("avg", numpy.mean)
    stats.register("max", numpy.max)

    # define the hall-of-fame object:
    hofs = []
    for _ in range(RUNS):
        hofs.append(tools.HallOfFame(HALL_OF_FAME_SIZE))

    # perform the Genetic Algorithm flow with elitism:
    logbooks = elitism.eaSimpleWithElitism(POPULATION_SIZE, toolbox, cxpb=P_CROSSOVER, mutpb=P_MUTATION, runs=RUNS,
                                              ngen=MAX_GENERATIONS, stats=stats, halloffame=hofs, verbose=False)

    printResult(hofs, logbooks, RUNS)


if __name__ == "__main__":
    main()
