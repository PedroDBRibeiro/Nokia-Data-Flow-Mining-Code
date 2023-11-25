from cec2019comp100digit import cec2019comp100digit
import numpy as np
from jmetal.core.problem import FloatProblem, S


class CEC2019(FloatProblem):

    def __init__(self, function_type: int = 0, number_of_variables: int = 0):

        """ jMetal common structure """
        super(CEC2019, self).__init__()
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Rosetta Energy Unit']

        self.function_type = function_type
        self.number_of_variables = number_of_variables

        self.set_bounds()

        self.benchmark = cec2019comp100digit
        self.benchmark.init(function_type, self.number_of_variables)

    def evaluate(self, solution: S) -> S:
        sol = np.array(solution.variables)
        fitness = self.benchmark.eval(sol)
        solution.objectives[0] = fitness

    def free_resources(self):
        self.benchmark.end()

    def get_name(self) -> str:
        return "CEC_2019_F" + str(self.function_type)

    '''
    1 Storn's Chebyshev Polynomial Fitting Problem 1 9 [-8192, 8192]
    2 Inverse Hilbert Matrix Problem 1 16 [-16384, 16384]
    3 Lennard-Jones Minimum Energy Cluster 1 18 [-4,4]
    4 Rastrigin’s Function 1 10 [-100,100]
    5 Griewangk’s Function 1 10 [-100,100]
    6 Weierstrass Function 1 10 [-100,100]
    7 Modified Schwefel’s Function 1 10 [-100,100]
    8 Expanded Schaffer’s F6 Function 1 10 [-100,100]
    9 Happy Cat Function 1 10 [-100,100]
    10 Ackley Function 1 10 [-100,100]
    '''

    def set_bounds(self):
        if self.function_type == 1:
            if self.number_of_variables == 0: self.number_of_variables = 9
            self.lower_bound = [-8192 for _ in range(self.number_of_variables)]
            self.upper_bound = [8192 for _ in range(self.number_of_variables)]
        elif self.function_type == 2:
            if self.number_of_variables == 0: self.number_of_variables = 16
            self.lower_bound = [-16384 for _ in range(self.number_of_variables)]
            self.upper_bound = [16384 for _ in range(self.number_of_variables)]
        elif self.function_type == 3:
            if self.number_of_variables == 0: self.number_of_variables = 18
            self.lower_bound = [-4 for _ in range(self.number_of_variables)]
            self.upper_bound = [4 for _ in range(self.number_of_variables)]
        elif self.function_type == 4:
            if self.number_of_variables == 0: self.number_of_variables = 10
            self.lower_bound = [-100 for _ in range(self.number_of_variables)]
            self.upper_bound = [100 for _ in range(self.number_of_variables)]
        elif self.function_type == 5:
            if self.number_of_variables == 0: self.number_of_variables = 10
            self.lower_bound = [-100 for _ in range(self.number_of_variables)]
            self.upper_bound = [100 for _ in range(self.number_of_variables)]
        elif self.function_type == 6:
            if self.number_of_variables == 0: self.number_of_variables = 10
            self.lower_bound = [-100 for _ in range(self.number_of_variables)]
            self.upper_bound = [100 for _ in range(self.number_of_variables)]
        elif self.function_type == 7:
            if self.number_of_variables == 0: self.number_of_variables = 10
            self.lower_bound = [-100 for _ in range(self.number_of_variables)]
            self.upper_bound = [100 for _ in range(self.number_of_variables)]
        elif self.function_type == 8:
            if self.number_of_variables == 0: self.number_of_variables = 10
            self.lower_bound = [-100 for _ in range(self.number_of_variables)]
            self.upper_bound = [100 for _ in range(self.number_of_variables)]
        elif self.function_type == 9:
            if self.number_of_variables == 0: self.number_of_variables = 10
            self.lower_bound = [-100 for _ in range(self.number_of_variables)]
            self.upper_bound = [100 for _ in range(self.number_of_variables)]
        elif self.function_type == 10:
            if self.number_of_variables == 0: self.number_of_variables = 10
            self.lower_bound = [-100 for _ in range(self.number_of_variables)]
            self.upper_bound = [100 for _ in range(self.number_of_variables)]
