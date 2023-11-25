import copy
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, List

from jmetal.core.problem import Problem
from jmetal.core.solution import Solution

import numpy as np

R = TypeVar('R')

"""
.. module:: generator
   :platform: Unix, Windows
   :synopsis: Population generators implementation.

.. moduleauthor:: Antonio Benítez-Hidalgo <antonio.b@uma.es>
"""


class Generator(Generic[R], ABC):

    @abstractmethod
    def new(self, problem: Problem = None) -> R:
        pass

    @abstractmethod
    def set_problem(self, problem: Problem = None):
        pass


class RandomGenerator(Generator):
    problem = None

    def new(self, problem: Problem = None):
        if problem is not None:
            self.problem = problem

        return self.problem.create_solution()

    def set_problem(self, problem: Problem = None):
        self.problem = problem


class InjectorGenerator(Generator):
    problem = None

    def __init__(self, solutions: List[Solution]):
        super(InjectorGenerator, self).__init__()
        self.population = []

        for solution in solutions:
            self.population.append(copy.deepcopy(solution))

    def new(self, problem: Problem = None):
        if len(self.population) > 0:
            # If we have more solutions to inject, return one from the list
            return self.population.pop()
        else:
            # Otherwise generate a new solution
            solution = problem.create_solution()

        return solution

    def set_problem(self, problem: Problem = None):
        self.problem = problem


class ArchiveInjector(Generator):

    def __init__(self, file_path: str = None, problem: Problem = None):
        self.file_path = file_path
        self.problem = problem
        self.list_of_solutions = None
        self.create_list_of_solutions()

    def create_list_of_solutions(self):
        self.list_of_solutions = []
        list_of_individuals = np.loadtxt(self.file_path, delimiter=',', dtype=float)

        for variables in list_of_individuals:
            sol = self.problem.create_solution()
            sol.variables = variables
            self.list_of_solutions.append(sol)

    def new(self, problem: Problem = None) -> Solution:
        if len(self.list_of_solutions) > 0:
            solution = self.list_of_solutions.pop()
        else:
            solution = self.problem.create_solution()

        return solution

    def set_problem(self, problem: Problem = None):
        self.problem = problem