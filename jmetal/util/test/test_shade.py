import unittest
from jmetal.problem.singleobjective.CEC2013 import CEC2013
from jmetal.util.solutions.generator import RandomGenerator
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.algorithm.singleobjective.shade import SHADE


class ShadeWithoutArchiveTestCases(unittest.TestCase):
    p = 0
    d = 30
    problem = CEC2013(p + 1, d)

    trials = 1

    max_evaluations = 10
    pop_size = 10

    population_generator = RandomGenerator()

    population_generator.set_problem(problem)

    termination_criteria = StoppingByEvaluations(max_evaluations)

    algorithm = SHADE(problem=problem, population_size=pop_size,
                      termination_criterion=termination_criteria,
                      population_generator=population_generator,
                      use_archive=False)

    def test_initial_state_memories(self):
        self.algorithm.init_progress()
        for mcr, mf in zip(self.algorithm.MemCR, self.algorithm.MemF):
            self.assertEqual(mcr, 0.5)
            self.assertEqual(mf, 0.5)


if __name__ == '__main__':
    unittest.main()
