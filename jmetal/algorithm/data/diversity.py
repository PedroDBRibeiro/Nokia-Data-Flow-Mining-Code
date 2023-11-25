from abc import abstractmethod
import numpy as np


class Diversity:

    @abstractmethod
    def get_data(self):
        pass


class DimensionWiseDiversity(Diversity):

    def __init__(self):
        self.max_diversity = 0
        self.curr_diversity = 0
        self.solutions = None
        self.xpl_history = []
        self.xpt_history = []

    def calc_median_sum(self, j):
        m_values = np.median(self.solutions, axis=0)
        total_sum = 0

        for x in self.solutions:
            total_sum += m_values[j] - x[j]

        total_sum = abs(total_sum)

        total = (1 / len(self.solutions)) * total_sum

        return total

    def calc_diversity(self, dimensionality):
        div_j = 0

        for j in range(dimensionality):
            div_j += self.calc_median_sum(j)

        div = div_j / dimensionality

        self.curr_diversity = div

    def calc_xpl(self): #exploration
        xpl = self.curr_diversity / self.max_diversity
        return xpl


    def calc_xpt(self): #exploitation
        xpt = abs(self.curr_diversity - self.max_diversity) / self.max_diversity
        return xpt

    def update(self, evaluations: int, solutions: list, number_of_dimensions: int):
        evaluations = evaluations
        self.solutions = np.array([sol.variables for sol in solutions])

        self.calc_diversity(number_of_dimensions)

        if evaluations / len(self.solutions) == 1:  # First Generation
            self.max_diversity = self.curr_diversity

        if self.curr_diversity > self.max_diversity:
            self.max_diversity = self.curr_diversity

        self.xpl_history.append(self.calc_xpl())
        self.xpt_history.append(self.calc_xpt())

    def get_data(self):
        return self.xpl_history[-1], self.xpt_history[-1]

    def get_full_data(self):
        return self.xpl_history, self.xpt_history
