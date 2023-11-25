import unittest
from math import inf

import numpy as np

from jmetal.algorithm.singleobjective.particle_swarm_algorithm import ParticleSwarmAlgorithm
from jmetal.core.solution import FloatSolution
from jmetal.problem.singleobjective.CEC2013 import CEC2013
from jmetal.util.solutions.generator import RandomGenerator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.operator.boundary_correction import LimitCorrection


class PSOTestCase(unittest.TestCase):

    def setUp(self) -> None:
        self.swarm_size = 10
        self.max_evaluations = 1000

        self.problem = CEC2013(1, 10)
        self.termination_criterion = StoppingByEvaluations(self.max_evaluations)
        self.swarm_generator = RandomGenerator()
        self.bound_correction = LimitCorrection()
        self.pso = ParticleSwarmAlgorithm(problem=self.problem,
                                          swarm_size=self.swarm_size,
                                          swarm_generator=self.swarm_generator,
                                          termination_criterion=self.termination_criterion,
                                          v_max=100, v_min=-100, c1=1.19315, c2=1.19315, W=0.72)
        self.pso.velocity_bound_correction = self.bound_correction
        self.pso.solutions = self.pso.create_initial_solutions()
        self.pso.evaluate(self.pso.solutions)

    def test_swarm_initialization(self):
        self.assertIsNotNone(self.pso.solutions, msg="Swarm population not created.")
        self.assertEqual(len(self.pso.solutions), self.swarm_size, msg="Incorrect swarm population size.")

    def test_velocity_initialization(self):
        self.pso.initialize_velocity(self.pso.solutions)

        velocities = []
        for particle in self.pso.solutions:
            velocities.append(particle.attributes['velocity'])

        self.assertEqual(self.swarm_size, len(velocities), msg="Velocities not initialized.")

    def test_pbest_initialization(self):
        self.pso.initialize_particle_best(self.pso.solutions)

        pbest_var = []
        pbest_obj = []

        for particle in self.pso.solutions:
            pbest_var.append(particle.attributes['pbest_variables'])
            self.assertEqual(pbest_var[-1], particle.variables, "Incorrect pBest variables initialization.")

            pbest_obj.append(particle.attributes['pbest_objective'])
            self.assertEqual(pbest_obj[-1], particle.objectives[0], "Incorrect pBest objective initialization.")

        self.assertEqual(len(pbest_obj), self.swarm_size, msg="pBest objectives are incorrect.")
        self.assertEqual(len(pbest_var), self.swarm_size, msg="pBest variables are incorrect.")

    def test_pbest_update(self):
        self.pso.initialize_particle_best(self.pso.solutions)

        for p in self.pso.solutions:
            p.variables = np.full(self.swarm_size, +inf)
            p.objectives = [-inf]

        self.pso.update_particle_best(self.pso.solutions)

        for p in self.pso.solutions:
            np.testing.assert_array_equal(np.full(self.swarm_size, +inf), p.attributes['pbest_variables'])
            self.assertEqual(-inf, p.attributes['pbest_objective'], msg="pBest objective not updated.")

    def test_gbest_initialization(self):
        self.pso.initialize_global_best(self.pso.solutions)

        for particle in self.pso.solutions:
            self.assertLessEqual(self.pso.gBest.objectives[0], particle.objectives[0])

    def test_gbest_update(self):
        self.pso.initialize_global_best(self.pso.solutions)
        self.pso.solutions[-1].objectives[0] = -inf
        self.pso.update_global_best(self.pso.solutions)

        self.assertEqual(self.pso.gBest.objectives[0], -inf, msg="gBest was not correctly updated!")

    def test_velocity_update(self):
        particle = FloatSolution(lower_bound=np.full(3, -100, dtype=float),
                                 upper_bound=np.full(3, 100, dtype=float),
                                 number_of_objectives=1,
                                 number_of_constraints=0)
        particle.objectives[0] = +inf
        particle.variables = np.full(3, 25, dtype=float)
        particle.attributes['velocity'] = np.full(3, 2, dtype=float)
        particle.attributes['pbest_variables'] = np.full(3, 15, dtype=float)
        particle.attributes['pbest_objective'] = -inf

        best_particle = FloatSolution(lower_bound=np.full(3, -100, dtype=float),
                                      upper_bound=np.full(3, 100, dtype=float),
                                      number_of_objectives=1,
                                      number_of_constraints=0)
        best_particle.objectives[0] = -inf
        best_particle.variables = np.full(3, 5, dtype=float)

        self.pso.gBest = best_particle
        self.pso.c1 = 1
        self.pso.c2 = 1
        self.pso.W = 0.5
        self.pso.solutions = [particle]
        self.pso.set_seed(1)
        self.pso.update_velocity(swarm=self.pso.solutions)

        updated_velocity = self.pso.solutions[0].attributes['velocity']

        np.testing.assert_almost_equal([-17.29231717, -11.739126, -12.944172],
                                       updated_velocity, decimal=4, err_msg="Velocity update is wrong.")

    def test_position_update(self):
        particle = FloatSolution(lower_bound=np.full(3, -100, dtype=float),
                                 upper_bound=np.full(3, 100, dtype=float),
                                 number_of_objectives=1,
                                 number_of_constraints=0)
        particle.objectives[0] = +inf
        particle.variables = np.full(3, 25, dtype=float)
        particle.attributes['velocity'] = np.full(3, 2, dtype=float)
        particle.attributes['pbest_variables'] = np.full(3, 15, dtype=float)
        particle.attributes['pbest_objective'] = -inf

        self.pso.solutions = [particle]
        self.pso.update_position(self.pso.solutions)

        updated_particle = self.pso.solutions[-1]

        np.testing.assert_almost_equal(updated_particle.variables, np.full(3, 27), err_msg="Wrong position update.")


if __name__ == '__main__':
    unittest.main()
