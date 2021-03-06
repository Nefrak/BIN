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
FILE_PATH = "sudoku.yaml"
PRINT_SUDOKU = True
SHOW_CONV_STAT = True
SHOW_BOX_STAT = True

# Genetic Algorithm constants:
POPULATION_SIZE = 100
P_CROSSOVER = 0
P_MUTATION = 0.7   
P_M_SWITCH = 0.1   
P_M_CONFLICT_SWITCH = 0.2 
P_M_SHIFT = 0.01     
RUNS = 15
MAX_GENERATIONS = 1000
HALL_OF_FAME_SIZE = 5

# set the random seed:
RANDOM_SEED = 123
random.seed(RANDOM_SEED)

#results
SAVEFILE = 'experiments/result2'

toolbox = base.Toolbox()

# create the graph coloring problem instance to be used:
gcp = graphs.GraphColoringProblem(FILE_PATH, HARD_CONSTRAINT_PENALTY)

# define a single objective, maximizing fitness strategy:
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))

# create the Individual class based on list:
creator.create("Individual", list, fitness=creator.FitnessMin)

# create an operator that randomly returns an integer in the range of participating colors:
toolbox.register("Integers", random.randint, 1, 9)

def getUnusedNumbers(block):
    num = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    for i in block:
        if i != 0:
            num.remove(i)
    return num

#creating new individual (no duplicit numbers allowed)
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
def allMutations(individual, mutateSwitchChance, mutateConflictChance, mutateShiftChance):
    for index in range(len(individual)):
        prob = random.random()
        if prob < mutateSwitchChance:
            individual = mutateSwitchNodes(index, individual)
        if prob < mutateConflictChance:
            individual = mutateSwitchConflictNodes(index, individual)
        if prob < mutateShiftChance:
            individual = mutateShiftBlock(index, individual)
    return individual

#custom genetic functions
def mutateSwitchNodes(index, individual):
    blockIndex = gcp.getBlock(index)
    confIndex = random.randrange(9 * blockIndex, 9 * (blockIndex + 1))
    isFixed = gcp.isFixed(index) or gcp.isFixed(confIndex)
    if not isFixed and confIndex != -1:
        savedColor = individual[index]
        individual[index] = individual[confIndex]
        individual[confIndex] = savedColor
    return individual

#mutating by switching conflict nodes
def mutateSwitchConflictNodes(index, individual):
    confIndex = gcp.isInViolationInBlock(index, individual)
    isFixed = gcp.isFixed(index) or gcp.isFixed(confIndex)
    if not isFixed and confIndex != -1:
        savedColor = individual[index]
        individual[index] = individual[confIndex]
        individual[confIndex] = savedColor
    return individual

# check if number is fixed (non mutable)
def isFixed(index, indexList):
        if index >= 0:
            for i in indexList:
                    if i == index:
                        return True
        return False

# rotating given block (then returning fixed numbers on their spot)
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

#mutating by shifting block with node
def mutateShiftBlock(index, individual):
    blockIndex = gcp.getBlock(index)
    oriBlock = individual[blockIndex * 9:(blockIndex + 1) * 9]
    indexList = [(i - blockIndex * 9) for i in range(blockIndex * 9, (blockIndex + 1) * 9) if gcp.isFixed(i)]
    newBlock = blockRotation(oriBlock, indexList)
    individual[blockIndex * 9:(blockIndex + 1) * 9] = newBlock
    return individual

#crossing one point on subgraphs
def crossOnePoint(ind1, ind2):
    crossPoint = random.randrange(1, 9)
    newind1 = ind1[0:crossPoint * 9].copy()
    ind1[0:crossPoint * 9] = ind2[0:crossPoint * 9]
    ind2[0:crossPoint * 9] = newind1
    return (ind1, ind2)

#crossing two point on subgraphs
def crossTwoPoints(ind1, ind2):
    crossPoint1 = random.randrange(1, 7)
    crossPoint2 = random.randrange(crossPoint1 + 1, 9)
    newind1 = ind1[crossPoint1 * 9:crossPoint2 * 9].copy()
    ind1[crossPoint1 * 9:crossPoint2 * 9] = ind2[crossPoint1 * 9:crossPoint2 * 9]
    ind2[crossPoint1 * 9:crossPoint2 * 9] = newind1
    return (ind1, ind2)

toolbox.register("evaluate", getCost)

# genetic operators:
toolbox.register("select", tools.selTournament, tournsize=2)
#toolbox.register("select", tools.selBest)

#toolbox.register("mate", crossOnePoint)
toolbox.register("mate", crossTwoPoints)

toolbox.register("mutate", allMutations, mutateSwitchChance=P_M_SWITCH, mutateConflictChance=P_M_CONFLICT_SWITCH, 
mutateShiftChance=P_M_SHIFT)

# show statistics
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

# show boxplot
def showBox(logbooks, runs):
    lastEval = [logbooks[0].select("avg")[-1]]
    lastEvalMin = [logbooks[0].select("min")[-1]]
    for i in range(1, runs):
       lastEval.append(logbooks[i].select("avg")[-1])
       lastEvalMin.append(logbooks[i].select("min")[-1])

    numpy.save(SAVEFILE, numpy.array(lastEvalMin))

    plt.figure(2)
    plt.boxplot([lastEval, lastEvalMin], labels=['avg', 'min'])
    plt.xlabel('Fitness')
    plt.ylabel('Outputs')
    plt.title('Boxplot')

#print final sudoku
def printSudoku(Sudoku):
    for row in range(9):
        print("")
        for subBlock in range(0, 3):
            for number in range(0, 3):
                position = number + subBlock * 27 + 3 * row
                print("|", end =" ")
                print(Sudoku[position], end =" ")
        print("|", end =" ")

# show results
def printResult(hofs, logbooks, runs):
    # print info for best solution found:
    best = hofs[0].items[0]
    for hof in hofs:
        if best.fitness.values[0] > hof.items[0].fitness.values[0]:
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
    if PRINT_SUDOKU:
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
