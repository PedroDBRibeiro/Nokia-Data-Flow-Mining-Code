import copy
import time
import random
from typing import TypeVar, List

import numpy as np

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm, Algorithm
from jmetal.core.problem import Problem
from jmetal.util.solutions.evaluator import Evaluator
from jmetal.util.solutions.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')


class DifferentialEvolution(EvolutionaryAlgorithm[S, R]):

    def __init__(self,
                 problem: Problem[S],
                 population_size: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator):

        super(DifferentialEvolution, self).__init__(problem=problem,
                                                    population_size=population_size,
                                                    offspring_population_size=population_size)

        self.population_evaluator = population_evaluator
        self.population_generator = population_generator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.crossover_operator = None
        self.selection_operator = None

    def selection(self, population: List[S]) -> np.ndarray:
        mating_pool = np.empty(len(population), dtype=np.ndarray)

        for i in range(self.population_size):
            self.selection_operator.set_index_to_exclude(i)
            selected_solutions = self.selection_operator.execute(self.solutions)
            mating_pool[i] = selected_solutions

        return mating_pool

    def reproduction(self, mating_pool: np.ndarray) -> List[S]:
        offspring_population = []

        for i, solution in enumerate(self.solutions):
            self.crossover_operator.current_individual = solution
            parents = mating_pool[i]
            offspring_population.append(self.crossover_operator.execute(parents))

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        tmp_list = []
        for solution1, solution2 in zip(self.solutions, offspring_population):
            if solution2.objectives[0] <= solution1.objectives[0]:
                tmp_list.append(solution2)
            else:
                tmp_list.append(solution1)

        return tmp_list

    def create_initial_solutions(self) -> List[S]:
        return [self.population_generator.new(self.problem) for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_result(self) -> List[S]:
        return self.solutions

    def get_best_solution(self):
        best_solution = sorted(self.solutions, key=lambda s: s.objectives[0])[0]
        return best_solution

    def get_name(self) -> str:
        return 'DE'

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.solutions,
                'COMPUTING_TIME': time.time() - self.start_computing_time}