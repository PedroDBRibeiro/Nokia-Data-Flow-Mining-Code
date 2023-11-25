from cec2013lsgo.cec2013 import Benchmark
import numpy as np
from jmetal.core.problem import FloatProblem, S


class CEC2013LSGO(FloatProblem):

    def __init__(self, function_type: int = 0, number_of_variables: int = 1000):

        super(CEC2013LSGO, self).__init__()
        self.number_of_objectives = 1
        self.number_of_constraints = 0
        self.number_of_variables = number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Fitness']

        self.function_type = function_type

        self.benchmark = Benchmark()

        info = self.benchmark.get_info(self.function_type)
        self.lower_bound = [info['lower'] for _ in range(self.number_of_variables)]
        self.upper_bound = [info['upper'] for _ in range(self.number_of_variables)]
        self.evaluator = self.benchmark.get_function(self.function_type)

    def evaluate(self, solution: S) -> S:
        sol = np.array(solution.variables)
        solution.objectives[0] = self.evaluator(sol)

        return solution

    def get_name(self) -> str:
        return "CEC_2013LSGO_F"+str(self.function_type)
