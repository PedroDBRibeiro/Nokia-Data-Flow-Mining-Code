import warnings
from typing import TypeVar, List

import pygmo

from jmetal.algorithm.singleobjective.differential_evolution import DifferentialEvolution
from jmetal.config import store
from jmetal.core.problem import Problem
from jmetal.operator.crossover import DifferentialEvolutionCrossover
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.density_estimator import CrowdingDistance
from jmetal.util.ranking import FastNonDominatedRanking
from jmetal.util.replacement import RankingAndDensityEstimatorReplacement, RemovalPolicyType
from jmetal.util.solutions.evaluator import Evaluator
from jmetal.util.solutions.generator import Generator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')


class DifferentialEvolutionMultiObjective(DifferentialEvolution):

    def __init__(self, problem: Problem[S], population_size: int,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 population_generator: Generator = store.default_generator,
                 population_evaluator: Evaluator = store.default_evaluator,
                 crossover_operator: DifferentialEvolutionCrossover = None,
                 selection_operator: DifferentialEvolutionSelection = None):

        super().__init__(problem=problem,
                         population_size=population_size)

        self.selection_operator = selection_operator
        self.crossover_operator = crossover_operator
        self.population_evaluator = population_evaluator
        self.population_generator = population_generator

        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        self.pool_of_solutions = []

        if crossover_operator is None:
            warnings.warn("Default crossover operator with CR = 0.9 and F = 0.5 parameters.")
            self.crossover_operator = DifferentialEvolutionCrossover(0.9, [0.5], 0.0)
        else:
            self.crossover_operator = crossover_operator

        if selection_operator is None:
            warnings.warn("Default selection operator will be used.")
            self.selection_operator = DifferentialEvolutionSelection()
        else:
            self.selection_operator = selection_operator

    def selection(self, population: List[S], index_to_exclude: int =None) -> List[S]:

        self.selection_operator.set_index_to_exclude(index_to_exclude)
        selected_solutions = self.selection_operator.execute(population)

        return selected_solutions

    def replacement(self, population: List[S], offspring_population: List[S]) -> List[S]:
        ranking = FastNonDominatedRanking()
        density_estimator = CrowdingDistance()

        r = RankingAndDensityEstimatorReplacement(ranking, density_estimator, RemovalPolicyType.ONE_SHOT)
        solutions = r.replace(population, offspring_population)

        return solutions

    def reproduction(self, mating_pool: List[S], solution_index: int = None) -> List[S]:
        offspring_population = []
        self.crossover_operator.current_individual = self.solutions[solution_index]
        offspring_population.append(self.crossover_operator.execute(mating_pool)[0])

        return offspring_population

    def step(self):
        self.pool_of_solutions = list(self.solutions)

        for j in range(self.population_size):
            mating_individuals = self.selection(self.pool_of_solutions, j)
            trial_solution = self.reproduction(mating_individuals, j)[0]
            trial_solution = self.evaluate([trial_solution])[-1]

            if pygmo.pareto_dominance(trial_solution.objectives, self.solutions[j].objectives):
                self.pool_of_solutions[j] = trial_solution
            elif pygmo.pareto_dominance(self.solutions[j].objectives, trial_solution.objectives):
                pass
            else:
                self.pool_of_solutions.append(trial_solution)

        self.solutions = self.replacement(self.pool_of_solutions[:self.population_size], self.pool_of_solutions[self.population_size:])

    def get_result(self) -> R:
        return self.solutions

    def get_name(self) -> str:
        return 'DEMO'

    @property
    def label(self) -> str:
        return f'{self.get_name()}.{self.problem.get_name()}'
