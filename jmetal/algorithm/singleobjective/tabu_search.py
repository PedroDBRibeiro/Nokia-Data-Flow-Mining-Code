import copy
import random
import threading
import time
from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Mutation
from jmetal.core.problem import Problem
from jmetal.core.solution import Solution
from jmetal.util.solutions.comparator import Comparator
from jmetal.util.termination_criterion import TerminationCriterion


S = TypeVar('S')
R = TypeVar('R')

class TabuSearch(Algorithm[S, R], threading.Thread):

    def __init__(self,
                 problem: Problem[S],
                 mutation: Mutation,
                 tabu_size: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 comparator: Comparator = store.default_comparator):
        
        super(TabuSearch, self).__init__()
        self.comparator = comparator
        self.problem = problem
        self.mutation = mutation

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.neighbors_value = 10
        self.tabu_size = tabu_size
        self.tabu_list = []


    def create_initial_solutions(self) -> List[S]:
        self.solutions.append(self.problem.create_solution())
        return self.solutions

    def evaluate(self, solutions: List[S]) -> List[S]:
        return [self.problem.evaluate(solutions[0])]

    def generate_neighbors(self, solution):
        neighbors = []
        for _ in range(self.neighbors_value):  # Generate neighboring solutions
            mutated_solution = copy.deepcopy(self.solutions[0])
            mutated_solution: Solution = self.mutation.execute(mutated_solution)
            mutated_solution = self.evaluate([mutated_solution])[0]
            neighbors.append(mutated_solution)   
        return neighbors


    def step(self) -> None:
            neighbors = self.generate_neighbors(self.solutions[0])
            best_neighbor = min(neighbors, key=lambda x: x.objectives[0])

            result = self.comparator.compare(best_neighbor, self.solutions[0])
            if result == 1 and best_neighbor not in self.tabu_list:
                    self.solutions[0] = best_neighbor
                    self.tabu_list.append(best_neighbor)

                    if len(self.tabu_list) > self.tabu_size:
                        self.tabu_list.pop(0)


    def init_progress(self) -> None:
        self.evaluations = 0

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def update_progress(self) -> None:
        self.evaluations += 1
        observable_data = self.get_observable_data()
        self.observable.notify_all(**observable_data)

    def get_result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return 'Tabu Search'

    def get_observable_data(self) -> dict:
        ctime = time.time() - self.start_computing_time
        return {'PROBLEM': self.problem, 'EVALUATIONS': self.evaluations, 'SOLUTIONS': self.get_result(),
                'COMPUTING_TIME': ctime}