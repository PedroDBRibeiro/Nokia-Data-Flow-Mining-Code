import copy
import time
from typing import TypeVar, List

import numpy as np

from jmetal.algorithm.singleobjective.differential_evolution import DifferentialEvolution
from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.operator.boundary_correction import JadeCorrection
from jmetal.operator.crossover import DifferentialEvolutionCurrentCrossover
from jmetal.operator.selection import PBestSelectionWithoutArchive, PBestSelectionWithArchive
from jmetal.util.solutions.evaluator import Evaluator
from jmetal.util.solutions.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')


class SHADE(DifferentialEvolution):

    def __init__(self, problem: Problem[S], 
                 population_size: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 H: int = 100,
                 use_archive: bool = False):

        super().__init__(problem, population_size, termination_criterion,
                         population_generator, population_evaluator)

        self.use_archive = use_archive
        if self.use_archive:
            self.selection_operator = PBestSelectionWithArchive()
        else:
            self.selection_operator = PBestSelectionWithoutArchive()
        self.crossover_operator = DifferentialEvolutionCurrentCrossover(0.9, 0.5, 0.5)
        self.crossover_operator.boundary_correction = JadeCorrection()

        self.H = H
        self.MemF = np.ones(self.H) * .5
        self.MemCR = np.ones(self.H) * .5
        self.CR = np.zeros(self.population_size)
        self.F = np.zeros(self.population_size)
        self.K = 0
        self.solution_archive = []

    def init_progress(self) -> None:
        super().init_progress()

        self.MemF = np.ones(self.H) * .5
        self.MemCR = np.ones(self.H) * .5
        self.CR = np.zeros(self.population_size)
        self.F = np.zeros(self.population_size)
        self.K = 0
        self.solution_archive = []

    def update_parameters(self, SCR, SF, improvements: List):
        total = np.sum(improvements)
        assert total > 0
        weights = improvements / total

        new_F = np.sum(weights * SF * SF) / np.sum(weights * SF)
        new_F = np.clip(new_F, 0, 1)
        new_CR = np.sum(weights * SCR)
        new_CR = np.clip(new_CR, 0, 1)

        self.MemF[self.K] = new_F
        self.MemCR[self.K] = new_CR

        self.K = (self.K + 1) % self.H

    def selection(self, population: List[S]) -> np.ndarray:
        mating_pool = np.empty(len(population), dtype=np.ndarray)

        for i in range(self.population_size):
            if self.use_archive:
                self.selection_operator.set_index_to_exclude(i)
                selected_solutions = self.selection_operator.execute([self.solutions, self.solution_archive])
            else:
                self.selection_operator.set_index_to_exclude(i)
                selected_solutions = self.selection_operator.execute([self.solutions, []])

            mating_pool[i] = selected_solutions

        return mating_pool

    def reproduction(self, mating_pool: np.ndarray) -> List[S]:
        offspring_population = []

        for i, solution in enumerate(self.solutions):
            index_H = np.random.randint(0, self.H)
            meanF = self.MemF[index_H]
            meanCR = self.MemCR[index_H]

            Fi = np.random.normal(meanF, 0.1)
            CRi = np.random.normal(meanCR, 0.1)

            self.crossover_operator.current_individual = solution
            self.crossover_operator.CR = CRi
            self.crossover_operator.F = Fi

            self.CR[i] = self.crossover_operator.CR
            self.F[i] = self.crossover_operator.F

            parents = mating_pool[i]
            trial = self.crossover_operator.execute(parents)
            trial.attributes['CR'] = self.crossover_operator.CR
            trial.attributes['F'] = self.crossover_operator.F
            trial.attributes['parent_idx'] = i
            offspring_population.append(trial)

        return offspring_population

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        final_offspring = []
        SCR = []
        SF = []
        parameter_weights = []

        for i, (parent, trial) in enumerate(zip(self.solutions, offspring_population)):
            if trial.objectives[0] < parent.objectives[0]:
                final_offspring.append(trial)
                SCR.append(self.CR[i])
                SF.append(self.F[i])
                parameter_weights.append(parent.objectives[0] - trial.objectives[0])

                if self.use_archive:
                    self.update_archive(parent)
            else:
                final_offspring.append(parent)

        if len(SCR) > 0 and len(SF) > 0:
            self.update_parameters(SCR=SCR, SF=SF, improvements=parameter_weights)

        return final_offspring

    def update_archive(self, solution: S):

        if len(self.solution_archive) < self.population_size:
            self.solution_archive.append(copy.deepcopy(solution))
        else:
            random_replacement = np.random.randint(0, self.population_size)
            self.solution_archive[random_replacement] = copy.deepcopy(solution)

    def get_name(self) -> str:
        return 'SHADE'

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.solutions,
                'COMPUTING_TIME': time.time() - self.start_computing_time}