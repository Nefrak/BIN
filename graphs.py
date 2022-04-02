import networkx as nx
import yaml
import matplotlib.pyplot as plt
import numpy as np

class GraphColoringProblem:
    """This class encapsulates the Graph Coloring problem
    """

    def __init__(self, filePath, hardConstraintPenalty):
        """
        :param graph: a NetworkX graph to be colored
        :param hardConstraintPenalty: penalty for hard constraint (coloring violation)
        """

        # initialize instance variables:
        if filePath != "":
            self.graph, sudoku = self.getGraph(filePath)

        self.fixed_indexes = []
        self.sudoku = [0] * len(sudoku) * len(sudoku[0])

        for x in range(len(sudoku)):
            for y in range(len(sudoku[x])):
                if sudoku[x][y] != 0:
                    if y < 3:
                        self.fixed_indexes.append(x * 3 + y)
                        self.sudoku[x * 3 + y] = sudoku[x][y]
                    elif y < 6:
                        self.fixed_indexes.append(24 + x * 3 + y)
                        self.sudoku[24 + x * 3 + y] = sudoku[x][y]
                    elif y < 9:
                        self.fixed_indexes.append(48 + x * 3 + y)
                        self.sudoku[48 + x * 3 + y] = sudoku[x][y]

        self.hardConstraintPenalty = hardConstraintPenalty

        # a list of the nodes in the graph:
        self.nodeList = list(self.graph.nodes)

        # adjacency matrix of the nodes -
        # matrix[i,j] equals '1' if nodes i and j are connected, or '0' otherwise:
        self.adjMatrix = nx.adjacency_matrix(self.graph).todense()

    def __len__(self):
        """
        :return: the number of nodes in the graph
        """
        return nx.number_of_nodes(self.graph)

    def getGraph(self, file):
        graph = nx.Graph()
        sudoku = []
        nodes = 81

        with open(file) as f:
            docs = yaml.load_all(f, Loader=yaml.FullLoader)
            for doc in docs:
                for k, v in doc.items():
                    if k == "Sudoku":
                        sudoku = v

        for i in range(nodes):
            graph.add_node(i)
        #connecting field
        for i in range(1, int(nodes/9) + 1):
            for k in range((i - 1) * 9, i * 9):
                for n in range(k + 1, i * 9):
                    graph.add_edge(k, n)
        #connecting collumn
        for k in range(27, nodes + 1, 27):
            for n in range(k-27, k):
                for m in range(n + 3, k, 3):
                    graph.add_edge(n, m)
        #connecting row
        for k in range(27, nodes + 1, 27):
            for l in range(k-27, k, 3):
                for n in range(l, l + 3):
                    for m in range(n + 1, l + 3):
                        graph.add_edge(n, m)
                    if l + 27 + 3 <= nodes + 1:
                        for m in range(l + 27, l + 27 + 3):
                            graph.add_edge(n, m)
                    if l + 2 * 27 + 3 <= nodes + 1:
                        for m in range(l + 2 * 27, l + 2 * 27 + 3):
                            graph.add_edge(n, m)
            
        return graph, sudoku

    def getCost(self, colorArrangement):
        """
        Calculates the cost of the suggested color arrangement
        :param colorArrangement: a list of integers representing the suggested color arrangement for the nodes,
        one color per node in the graph
        :return: Calculated cost of the arrangement.
        """

        return self.hardConstraintPenalty * self.getViolationsCount(colorArrangement) + self.getNumberOfColors(colorArrangement)

    def getViolationsCount(self, colorArrangement):
        """
        Calculates the number of violations in the given color arrangement. Each pair of interconnected nodes
        with the same color counts as one violation.
        :param colorArrangement: a list of integers representing the suggested color arrangement for the nodes,
        one color per node in the graph
        :return: the calculated value
        """

        if len(colorArrangement) != self.__len__():
            raise ValueError("size of color arrangement should be equal to ", self.__len__())

        violations = 0

        # iterate over every pair of nodes and find if they are adjacent AND share the same color:
        for i in range(len(colorArrangement)):
            for j in range(i + 1, len(colorArrangement)):

                if self.adjMatrix[i, j]:    # these are adjacent nodes
                    if colorArrangement[i] == colorArrangement[j]:
                        violations += 1

        return violations

    def getBlock(self, index):
        return index // 9

    def isInViolationInBlock(self, index, colorArrangement):
        block = self.getBlock(index)
        for i in range(block * 9, (block + 1) * 9):
                if i != index and colorArrangement[index] == colorArrangement[i]:
                    return i
        return -1

    def isInViolation(self, index, colorArrangement):
        for i in range(len(colorArrangement)):
                if i != index and colorArrangement[index] == colorArrangement[i]:
                    return i
        return -1

    def isFixed(self, index):
        if index >= 0:
            for i in self.fixed_indexes:
                    if i == index:
                        return True
        return False

    def getNumberOfColors(self, colorArrangement):
        """
        returns the number of different colors in the suggested color arrangement
        :param colorArrangement: a list of integers representing the suggested color arrangement fpo the nodes,
        one color per node in the graph
        :return: number of different colors
        """
        return len(set(colorArrangement))

    def plotGraph(self, colorArrangement):
        """
        Plots the graph with the nodes colored according to the given color arrangement
        :param colorArrangement: a list of integers representing the suggested color arrangement fpo the nodes,
        one color per node in the graph
        """

        if len(colorArrangement) != self.__len__():
            raise ValueError("size of color list should be equal to ", self.__len__())

        # create a list of the unique colors in the arrangement:
        colorList = list(set(colorArrangement))

        # create the actual colors for the integers in the color list:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(colorList)))

        # iterate over the nodes, and give each one of them its corresponding color:
        colorMap = []
        for i in range(self.__len__()):
            color = colors[colorList.index(colorArrangement[i])]
            colorMap.append(color)

        # plot the nodes with their labels and matching colors:
        nx.draw_kamada_kawai(self.graph, node_color=colorMap, with_labels=True)
        #nx.draw_circular(self.graph, node_color=color_map, with_labels=True)

        return plt


# testing the class:
def main():
    # create a problem instance with petersen graph:
    gcp = GraphColoringProblem(nx.petersen_graph(), 10)

    # generate a random solution with up to 5 different colors:
    solution = np.random.randint(5, size=len(gcp))

    print("solution = ", solution)
    print("number of colors = ", gcp.getNumberOfColors(solution))
    print("Number of violations = ", gcp.getViolationsCount(solution))
    print("Cost = ", gcp.getCost(solution))

    plot = gcp.plotGraph(solution)
    plot.show()


if __name__ == "__main__":
    main()

