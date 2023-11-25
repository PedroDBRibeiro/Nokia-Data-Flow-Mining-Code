from typing import List

from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.problem.singleobjective.CEC2013LSGO import CEC2013LSGO


class InitialPopulationCreator:

    def __init__(self, problem: Problem = None,
                 number_of_individuals: int = 100,
                 number_of_populations: int = 1):

        self.problem = problem
        self.number_of_individuals = number_of_individuals
        self.number_of_populations = number_of_populations
        self.populations = None

    def generate_solutions(self) -> List[Solution]:
        solutions = []

        for i in range(self.number_of_individuals):
            solutions.append(self.problem.create_solution())

        return solutions

    def create_populations(self) -> List:
        populations = []

        for i in range(self.number_of_populations):
            populations.append(self.generate_solutions())

        self.populations = populations

        return populations

    def export_populations_to_file(self, file_path) -> None:

        for i in range(self.number_of_populations):
            f = open(file_path + "/population_" + str(i) + ".txt", "w")
            for j in range(self.number_of_individuals):
                f.write(",".join(str(_) for _ in self.populations[i][j].variables) + "\n")
            f.close()


if __name__ == '__main__':
    problem_id = 15
    problem = CEC2013LSGO(problem_id)

    file_path = "/home/pnarloch/workspace/resources/populations/LSGO/function_" + str(problem_id)

    creator = InitialPopulationCreator(problem=problem, number_of_populations=30)
    creator.create_populations()
    creator.export_populations_to_file(file_path)
