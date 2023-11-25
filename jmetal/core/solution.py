from abc import ABC
from typing import List, Generic, TypeVar
from numpy import random, array, linalg, zeros, argmin, argsort, exp
from numpy.random import choice
import sys

BitSet = List[bool]
S = TypeVar('S')


class Solution(Generic[S], ABC):
    """ Class representing solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_constraints = number_of_constraints
        self.variables = [[] for _ in range(self.number_of_variables)]
        self.objectives = [0.0 for _ in range(self.number_of_objectives)]
        self.constraints = [0.0 for _ in range(self.number_of_constraints)]
        self.attributes = {}

    def __eq__(self, solution) -> bool:
        if isinstance(solution, self.__class__):
            return self.variables == solution.variables
        return False

    def __str__(self) -> str:
        return 'Solution(variables={},objectives={},constraints={})'.format(self.variables, self.objectives, self.constraints)


class BinarySolution(Solution[BitSet]):
    """ Class representing float solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super(BinarySolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = BinarySolution(
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution

    def get_total_number_of_bits(self) -> int:
        total = 0
        for var in self.variables:
            total += len(var)

        return total

    def get_binary_string(self) -> str:
        string = ""
        print(self.variables)
        for bit in self.variables[0]:
            string += '1' if bit else '0'
        return string


class FloatSolution(Solution[float]):
    """ Class representing float solutions """

    def __init__(self, lower_bound: List[float], upper_bound: List[float], number_of_objectives: int,
                 number_of_constraints: int = 0):
        super(FloatSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = FloatSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constraints)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        new_solution.attributes = self.attributes.copy()

        return new_solution

class Particle(FloatSolution):

    def __init__(self, ss):
        # self.number_of_variables = number_of_variables
        #ssNp = array(ss)
        self.lower_bound = array(ss)[:,0].tolist()
        self.upper_bound = array(ss)[:,1].tolist()
        super().__init__(lower_bound=self.lower_bound ,upper_bound=self.upper_bound,number_of_objectives=1)
        self.variables = []
        self.attributes = {}
        self.X = []
        self.V = []
        self.B = []
        self.B_discrete = None
        self.MarkedForRestart = False
        self.CalculatedFitness = sys.float_info.max
        self.FitnessDevStandard = sys.float_info.max
        self.CalculatedBestFitness = sys.float_info.max
        self.SinceLastLocalUpdate = 0

        self.DerivativeFitness = 0
        self.MagnitudeMovement = 0
        self.DistanceFromBest = sys.float_info.max
        self.CognitiveFactor = 2.
        self.SocialFactor = 2.
        self.Inertia = 0.5

        # support for PPSO
        self.MaxSpeedMultiplier = .25
        self.MinSpeedMultiplier = 0
        self.GammaInverter = 1

        self.cannot_move = False

        # used in the case of discrete optimization
        self._last_discrete_sample = None

    def can_move(self):
        if self.cannot_move:
            self.cannot_move = False
            return False
        else:
            return True

    def __repr__(self):
        return "<Particle %s>" % str(self.X)

    def __str__(self):
        return "\t".join(map(str, self.X))

    def _mark_for_restart(self):
        self.MarkedForRestart = True




class IntegerSolution(Solution[int]):
    """ Class representing integer solutions """

    def __init__(self, lower_bound: List[int], upper_bound: List[int], number_of_objectives: int,
                  number_of_constraints: int = 0):
        super(IntegerSolution, self).__init__(len(lower_bound), number_of_objectives, number_of_constraints)
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

    def __copy__(self):
        new_solution = IntegerSolution(
            self.lower_bound,
            self.upper_bound,
            self.number_of_objectives,
            self.number_of_constrains)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]
        new_solution.constraints = self.constraints[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution


class PermutationSolution(Solution):
    """ Class representing permutation solutions """

    def __init__(self, number_of_variables: int, number_of_objectives: int, number_of_constraints: int = 0):
        super(PermutationSolution, self).__init__(number_of_variables, number_of_objectives, number_of_constraints)

    def __copy__(self):
        new_solution = PermutationSolution(
            self.number_of_variables,
            self.number_of_objectives)
        new_solution.objectives = self.objectives[:]
        new_solution.variables = self.variables[:]

        new_solution.attributes = self.attributes.copy()

        return new_solution
