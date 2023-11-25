import copy
import random
import time
from typing import List

import numpy as np

from jmetal.config import store
from jmetal.core.algorithm import ParticleSwarmOptimization, R
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution, S
from jmetal.util.solutions import Evaluator, Generator
from jmetal.util.termination_criterion import TerminationCriterion
import jmetal.operator.boundary_correction as boundary


class ParticleSwarmAlgorithm(ParticleSwarmOptimization):

    def __init__(self, problem: Problem[S], swarm_size: int,
                 swarm_evaluator: Evaluator = store.default_evaluator,
                 swarm_generator: Generator = store.default_generator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 velocity_correction: boundary.Correction = None,
                 position_correction: boundary.Correction = None,
                 v_max: list = None,
                 v_min: list = None,
                 c1: float = None,
                 c2: float = None,
                 W: float = None):

        super().__init__(problem, swarm_size)
        self.v_max = v_max
        self.v_min = v_min
        self.W = W
        self.c1 = c1
        self.c2 = c2
        self.solutions = None
        self.gBest = None
        self.swarm_generator = swarm_generator
        self.swarm_evaluator = swarm_evaluator
        self.termination_criterion = termination_criterion
        self.velocity_bound_correction = velocity_correction
        self.position_bound_correction = position_correction

        self.observable.register(termination_criterion)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def create_initial_solutions(self) -> List[S]:
        return [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return self.swarm_evaluator.evaluate(solution_list=solution_list, problem=self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def get_result(self) -> R:
        return self.solutions

    def get_best_solution(self) -> S:
        return self.gBest

    def get_name(self) -> str:
        return "PSO"

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            velocity = np.zeros(self.problem.number_of_variables)
            particle.attributes['velocity'] = velocity

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['pbest_variables'] = particle.variables
            particle.attributes['pbest_objective'] = particle.objectives[0]

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        self.gBest = swarm[0]

        for particle in swarm:
            if particle.objectives[0] <= self.gBest.objectives[0]:
                self.gBest = copy.deepcopy(particle)

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            curr_velocity = particle.attributes['velocity']
            pBest = particle.attributes['pbest_variables']
            new_velocity = curr_velocity

            for idx, v in enumerate(curr_velocity):
                r1 = random.random()
                r2 = random.random()

                new_velocity[idx] = self.W * v + self.c1 * r1 * (pBest[idx] - particle.variables[idx]) \
                                    + self.c2 * r2 * (self.gBest.variables[idx] - particle.variables[idx])

            particle.attributes['velocity'] = new_velocity

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if particle.objectives[0] <= particle.attributes['pbest_objective']:
                particle.attributes['pbest_variables'] = particle.variables
                particle.attributes['pbest_objective'] = particle.objectives[0]

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        sorted_particles = np.argsort([_.objectives[0] for _ in swarm])
        particle = swarm[sorted_particles[0]]

        if particle.objectives[0] <= self.gBest.objectives[0]:
            self.gBest = copy.deepcopy(particle)

    def update_position(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.variables = particle.variables + particle.attributes['velocity']
            for idx, pos in enumerate(particle.variables):
                particle.variables[idx] = self.position_bound_correction.correct_boundaries(particle.upper_bound[idx],
                                                                                            particle.lower_bound[idx],
                                                                                            pos, pos)

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass

    def get_observable_data(self) -> dict:
        return {'PROBLEM': self.problem,
                'EVALUATIONS': self.evaluations,
                'SOLUTIONS': self.solutions,
                'GLOBAL_BEST': self.gBest,
                'COMPUTING_TIME': time.time() - self.start_computing_time}

    @property
    def label(self) -> str:
        return "PSO"
