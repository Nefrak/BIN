from deap import base
from deap import creator
from deap import tools

import random
import numpy
from collections import deque

import matplotlib.pyplot as plt
import seaborn as sns

import elitism
import graphs

# problem constants:
HARD_CONSTRAINT_PENALTY = 10  # the penalty factor for a hard-constraint violation
FILE_PATH = "params.yaml"
SHOW_GRAPH = True
SHOW_CONV_STAT = False
SHOW_BOX_STAT = False

# Genetic Algorithm constants:
POPULATION_SIZE = 200
P_CROSSOVER = 0.8  # probability for crossover
P_MUTATION = 0.8   # probability for mutating an individual
P_M_SWITCH = 0.7   # probability for performing switch in all mutations
P_M_CONFLICT = 0.7  # probability for performing switch conflicted nodes in all mutations
P_M_SHIFT = 0.8     # probability for performing shift of block in all mutations
RUNS = 4
MAX_GENERATIONS = 2500
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
toolbox.register("Integers", random.randint, 1, MAX_COLORS)

def getUnusedNumbers(block):
    num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in block:
        if i != 0:
            num.remove(i)
    return num

def createIndividual(individual):
    ind = individual(i for i in gcp.sudoku)
    for i in range(0,9):
        block = ind[i * 9:(i + 1) * 9]
        unusedNums = getUnusedNumbers(block)
        for n in range(len(block)):
            if block[n] == 0:
                ranNum = random.choice(unusedNums)
                unusedNums.remove(ranNum)
                block[n] = ranNum
        ind[i * 9:(i + 1) * 9] = block
    return ind

# create the individual operator to fill up an Individual instance:
#toolbox.register("individualCreator", tools.initRepeat, creator.Individual, toolbox.Integers, len(gcp))
toolbox.register("individualCreator", createIndividual, creator.Individual)

# create the population operator to generate a list of individuals:
toolbox.register("populationCreator", tools.initRepeat, list, toolbox.individualCreator)

# fitness calculation: cost of the suggested olution
def getCost(individual):
    return gcp.getCost(individual),  # return a tuple

#custom genetic functions
def allMutations(individual, mutateSwitchChance, mutateConflictChance, mutateShiftChance, indpb):
    prob = random.random()
    if prob < indpb:
        if prob < mutateSwitchChance:
            individual = mutateSwitchNodes(individual, indpb)
        if prob < mutateConflictChance:
            individual = mutateSwitchConflictNodes(individual, indpb)
        if prob < mutateShiftChance:
            individual = mutateShiftBlock(individual, indpb)
    return individual

#custom genetic functions
def mutateSwitchNodes(individual, indpb):
    for index in range(len(individual)):
        blockIndex = gcp.getBlock(index)
        confIndex = random.randrange(9 * blockIndex, 9 * (blockIndex + 1))
        isFixed = gcp.isFixed(index) and gcp.isFixed(confIndex)
        if not isFixed and confIndex != -1 and random.random() < indpb:
            savedColor = individual[index]
            individual[index] = individual[confIndex]
            individual[confIndex] = savedColor
    return individual

def mutateSwitchConflictNodes(individual, indpb):
    for index in range(len(individual)):
        confIndex = gcp.isInViolationInBlock(index, individual)
        isFixed = gcp.isFixed(index) and gcp.isFixed(confIndex)
        if not isFixed and confIndex != -1 and random.random() < indpb:
            savedColor = individual[index]
            individual[index] = individual[confIndex]
            individual[confIndex] = savedColor
    return individual

def isFixed(index, indexList):
        if index >= 0:
            for i in indexList:
                    if i == index:
                        return True
        return False

def blockRotation(oriBlock, indexList):
    if len(indexList) >= len(oriBlock):
        return oriBlock
    else:
        newBlock = deque(oriBlock)
        newBlock.rotate(1)
        newBlock = list(newBlock)
        n = len(newBlock) - 1
        while n > 0:
            if isFixed(n, indexList) and not isFixed(n - 1, indexList):
                savedColor = newBlock[0]
                newBlock[0] = newBlock[n]
                newBlock[n] = savedColor
                break
            n -= 1
        for i in range(len(newBlock) - 1):
            if isFixed(i, indexList):
                savedColor = newBlock[i]
                newBlock[i] = newBlock[i+1]
                newBlock[i+1] = savedColor
    return newBlock

#TODO protect fixed indexes gl
def mutateShiftBlock(individual, indpb):
    for index in range(len(individual)):
        blockIndex = gcp.getBlock(index)
        if random.random() < indpb:
            blockIndex = gcp.getBlock(index)
            oriBlock = individual[blockIndex * 9:(blockIndex + 1) * 9]
            indexList = [(i - blockIndex * 9) for i in range(blockIndex * 9, (blockIndex + 1) * 9) if gcp.isFixed(i)]
            newBlock = blockRotation(oriBlock, indexList)
            individual[blockIndex * 9:(blockIndex + 1) * 9] = newBlock
    return individual

def crossOnePoint(ind1, ind2):
    crossPoint = random.randrange(1, 9)
    newind1 = ind1.copy()
    ind1[0:crossPoint * 9] = ind2[0:crossPoint * 9]
    ind2[0:crossPoint * 9] = newind1[0:crossPoint * 9]
    return (ind1, ind2)

def crossTwoPoints(ind1, ind2):
    crossPoint1 = random.randrange(1, 7)
    crossPoint2 = random.randrange(crossPoint1 + 1, 9)
    newind1 = ind1.copy()
    ind1[crossPoint1 * 9:crossPoint2 * 9] = ind2[crossPoint1 * 9:crossPoint2 * 9]
    ind2[crossPoint1 * 9:crossPoint2 * 9] = newind1[crossPoint1 * 9:crossPoint2 * 9]
    return (ind1, ind2)

toolbox.register("evaluate", getCost)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
#toolbox.register("select", tools.selBest)

#toolbox.register("mate", crossOnePoint)
toolbox.register("mate", crossTwoPoints)

toolbox.register("mutate", allMutations, mutateSwitchChance=P_M_SWITCH, mutateConflictChance=P_M_CONFLICT, 
mutateShiftChance=P_M_SHIFT, indpb=1.0/len(gcp))
#toolbox.register("mutate", mutateSwitchNodes, indpb=1.0/len(gcp))
#toolbox.register("mutate", mutateShiftBlock, indpb=1.0/len(gcp))
#toolbox.register("mutate", mutateSwitchConflictNodes, indpb=1.0/len(gcp))

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

def printSudoku(Sudoku):
    for row in range(9):
        print("")
        for subBlock in range(0, 3):
            for number in range(0, 3):
                position = number + subBlock * 27 + 3 * row
                print("|", end =" ")
                print(Sudoku[position], end =" ")
        print("|", end =" ")

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
        printSudoku(best)

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
                                              ngen=MAX_GENERATIONS, sudoku=gcp.sudoku, stats=stats, halloffame=hofs, verbose=False)

    printResult(hofs, logbooks, RUNS)


if __name__ == "__main__":
    main()
