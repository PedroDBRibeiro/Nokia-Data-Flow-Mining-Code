import math
import random

import math

from jmetal.core.problem import BinaryProblem, FloatProblem, S
from jmetal.core.solution import BinarySolution, FloatSolution

"""
.. module:: unconstrained
   :platform: Unix, Windows
   :synopsis: Unconstrained test problems for single-objective optimization

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>, Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class OneMax(BinaryProblem):

    def __init__(self, number_of_bits: int = 256):
        super(OneMax, self).__init__()
        self.number_of_bits = number_of_bits
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Ones']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        counter_of_ones = 0
        for bits in solution.variables[0]:
            if bits:
                counter_of_ones += 1

        solution.objectives[0] = -1.0 * counter_of_ones

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]
        return new_solution

    def get_name(self) -> str:
        return 'OneMax'


class Sphere(FloatProblem):

    def __init__(self, number_of_variables: int = 10):
        super(Sphere, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for x in solution.variables:
            total += x * x

        solution.objectives[0] = total

        return solution

    def get_name(self) -> str:
        return 'Sphere'


class Rastrigin(FloatProblem):

    def __init__(self, number_of_variables: int = 10):
        super(Rastrigin, self).__init__()
        self.number_of_objectives = 1
        self.number_of_variables = number_of_variables
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5.12 for _ in range(number_of_variables)]
        self.upper_bound = [5.12 for _ in range(number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        a = 10.0
        result = a * solution.number_of_variables
        x = solution.variables

        for i in range(solution.number_of_variables):
            result += x[i] * x[i] - a * math.cos(2 * math.pi * x[i])

        solution.objectives[0] = result

        return solution

    def get_name(self) -> str:
        return 'Rastrigin'


class Rosenbrock(FloatProblem):

    def __init__(self):
        super().__init__()
        self.number_of_variables = 2
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        self.lower_bound = [-5 for _ in range(self.number_of_variables)]
        self.upper_bound = [10 for _ in range(self.number_of_variables)]

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def evaluate(self, solution: S) -> S:
        a = solution.variables[0]
        print(a)
        b = solution.variables[1]
        print(b)
        print(((1.0 - a) ** 2) + (100.0 * (b - (a ** 2)) ** 2))
        solution.objectives[0] = ((1.0 - a) ** 2) + (100.0 * (b - (a ** 2)) ** 2)

        return solution

    def get_name(self) -> str:
        return 'Banana Rosenbrok 2D'


class SubsetSum(BinaryProblem):

    def __init__(self, C: int, W: list):
        """ The goal is to find a subset S of W whose elements sum is closest to (without exceeding) C.

        :param C: Large integer.
        :param W: Set of non-negative integers."""
        super(SubsetSum, self).__init__()
        self.C = C
        self.W = W

        self.number_of_bits = len(self.W)
        self.number_of_objectives = 1
        self.number_of_variables = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MAXIMIZE]
        self.obj_labels = ['Sum']

    def evaluate(self, solution: BinarySolution) -> BinarySolution:
        total_sum = 0.0

        for index, bits in enumerate(solution.variables[0]):
            if bits:
                total_sum += self.W[index]

        if total_sum > self.C:
            total_sum = self.C - total_sum * 0.1

            if total_sum < 0.0:
                total_sum = 0.0

        solution.objectives[0] = -1.0 * total_sum

        return solution

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=1, number_of_objectives=1)
        new_solution.variables[0] = \
            [True if random.randint(0, 1) == 0 else False for _ in range(self.number_of_bits)]

        return new_solution

    def get_name(self) -> str:
        return 'Subset Sum'
