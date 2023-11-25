import logging
import math
import os
from pathlib import Path
from typing import List, TypeVar

import numpy as np
from tqdm import tqdm

from jmetal.core.observer import Observer
from jmetal.core.problem import DynamicProblem
from jmetal.core.quality_indicator import InvertedGenerationalDistance
from jmetal.lab.visualization import StreamingPlot, Plot
from jmetal.util.solutions import print_function_values_to_file

S = TypeVar('S')

LOGGER = logging.getLogger('jmetal')

"""
.. module:: observer
   :platform: Unix, Windows
   :synopsis: Implementation of algorithm's observers.

.. moduleauthor:: Antonio J. Nebro <antonio@lcc.uma.es>
"""


class ProgressBarObserver(Observer):

    def __init__(self, max: int) -> None:
        """ Show a smart progress meter with the number of evaluations and computing time.

        :param max: Number of expected iterations.
        :param desc: Prefix for the progressbar.
        """
        self.progress_bar = None
        self.progress = 0
        self.maxx = max

    def update(self, *args, **kwargs):
        if not self.progress_bar:
            self.progress_bar = tqdm(total=self.maxx, ascii=True, desc='Progress')

        evaluations = kwargs['EVALUATIONS']

        self.progress_bar.update(evaluations - self.progress)
        self.progress = evaluations

        if self.progress >= self.maxx:
            self.progress_bar.close()


class BasicObserver(Observer):

    def __init__(self, frequency: float = 1.0) -> None:
        """ Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency. """
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        computing_time = kwargs['COMPUTING_TIME']
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = solutions[0].objectives
            else:
                fitness = solutions.objectives

            LOGGER.info(
                'Evaluations: {} \n Best fitness: {} \n Computing time: {}'.format(
                    evaluations, fitness, computing_time
                )
            )


class AttributeObjectiveObserver(Observer):

    def __init__(self, frequency: float = 1.0) -> None:
        """ Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency. """
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        solutions = kwargs['SOLUTIONS']

        fitness = [_.attributes['function_score'] for _ in solutions]
        np_arr = np.sort(fitness)

        best_fitness = np_arr[0]

        LOGGER.info('Original Fitness: {}'.format(best_fitness))


class MechanismProbabilitiesObserver(Observer):

    def update(self, *args, **kwargs):
        d_prob = kwargs['DIVERSIFICATION_PERCENTAGE']
        i_prob = kwargs['INTENSIFICATION_PERCENTAGE']
        n_prob = kwargs['DO_NOTHING_PERCENTAGE']

        self.intensification_history.append(i_prob)
        self.diversfication_history.append(d_prob)
        self.do_nothing_history.append(n_prob)

    def __init__(self) -> None:
        super(MechanismProbabilitiesObserver, self).__init__()
        self.intensification_history = []
        self.diversfication_history = []
        self.do_nothing_history = []


class PrintObjectivesObserver(Observer):

    def __init__(self, frequency: float = 1.0) -> None:
        """ Show the number of evaluations, best fitness and computing time.

        :param frequency: Display frequency. """
        self.display_frequency = frequency
        self.fitness_history = []

    def update(self, *args, **kwargs):
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if (evaluations % self.display_frequency) == 0 and solutions:
            if type(solutions) == list:
                fitness = sorted(solutions, key=lambda s: s.objectives[0])[0].objectives[0]
            else:
                fitness = solutions.objectives

            self.fitness_history.append(fitness)

            LOGGER.info(
                'Evaluations: {}. fitness: {}'.format(
                    evaluations, fitness
                )
            )


class ParticleSwarmObserver(Observer):

    def __init__(self, frequency: float = 1.0) -> None:

        self.gBest_history = []
        self.display_frequency = frequency

    def update(self, *args, **kwargs):
        particles = kwargs['SOLUTIONS']
        evaluations = kwargs['EVALUATIONS']
        best_particle = kwargs['GLOBAL_BEST']

        mean_local_best = np.mean([_.attributes['pbest_objective'] for _ in particles])

        if (evaluations % self.display_frequency) == 0:
            self.gBest_history.append(best_particle.objectives[0])
            LOGGER.info(
                'Evaluatios: {}. Mean Local Best {}. Global Best: {}'.format(
                    evaluations, mean_local_best, best_particle.objectives[0]
                )
            )


class NoveltyDiversityObserver(Observer):

    def __init__(self):
        self.novelty_history = []

    def update(self, *args, **kwargs):
        solutions = kwargs['SOLUTIONS']
        evaluations = kwargs['EVALUATIONS']

        problem = kwargs['PROBLEM']

        mean_novelty = np.mean([_.objectives[1] for _ in solutions])

        self.novelty_history.append(mean_novelty)

        LOGGER.info('Evaluations: {}. Mean Novelty%: {}'.format(evaluations, self.novelty_history[-1]))


class DimensionWiseDiversityObserver(Observer):

    def __init__(self):
        self.max_diversity = 0
        self.curr_diversity = 0
        self.solutions = None
        self.xpl_history = []
        self.xpt_history = []

    def calc_median_sum(self, j):

        m_values = np.median(self.solutions, axis=0)
        total_sum = 0

        for x in self.solutions:
            total_sum += m_values[j] - x[j]

        total_sum = abs(total_sum)

        total = (1 / len(self.solutions)) * total_sum

        return total

    def calc_diversity(self, dimensionality):

        div_j = 0

        for j in range(dimensionality):
            div_j += self.calc_median_sum(j)

        div = div_j / dimensionality

        self.curr_diversity = div

    def calc_xpl(self):
        xpl = self.curr_diversity / self.max_diversity
        return xpl

    def calc_xpt(self):
        xpt = abs(self.curr_diversity - self.max_diversity) / self.max_diversity
        return xpt

    def update(self, *args, **kwargs):

        evaluations = kwargs['EVALUATIONS']
        self.solutions = np.array([sol.variables for sol in kwargs['SOLUTIONS']])

        problem = kwargs['PROBLEM']

        self.calc_diversity(problem.number_of_variables)

        if evaluations / len(self.solutions) == 1:  # First Generation
            self.max_diversity = self.curr_diversity

        if self.curr_diversity > self.max_diversity:
            self.max_diversity = self.curr_diversity

        self.xpl_history.append(self.calc_xpl())
        self.xpt_history.append(self.calc_xpt())

        return self.xpl_history[-1], self.xpt_history[-1]

        #LOGGER.info(
           # 'Evaluations: {}. XPL%: {}\t XPT%: {}'.format(evaluations, self.xpl_history[-1], self.xpt_history[-1])
        #)


class PrintDiversityObserver(Observer):

    def __init__(self):
        self.m_nmdf = 0
        self.diversity = []

    def update(self, *args, **kwargs):

        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        self.diversity.append(self.update_diversity(solutions))

        LOGGER.info(
            'Evaluations: {}. Populational Diversity: {}'.format(evaluations, self.diversity[-1])
        )

    def update_diversity(self, solutions):
        div = 0
        aux_2 = 0

        for a in range(0, len(solutions)):
            b = a + 1
            for i in range(b, len(solutions)):
                aux_1 = 0
                ind_a = solutions[a].variables
                ind_b = solutions[i].variables

                for d in range(0, len(ind_a)):
                    aux_1 = aux_1 + pow(ind_a[d] - ind_b[d], 2).real

                aux_1 = math.sqrt(aux_1).real
                aux_1 = (1 / len(ind_a)) * aux_1
                aux_1 = aux_1.real

                if b == i or aux_2 > aux_1:
                    aux_2 = aux_1

                div = div + (math.log(1.0 + aux_2)).real

                if self.m_nmdf < div:
                    self.m_nmdf = div

        return (div / self.m_nmdf).real


class WriteFrontToFileObserver(Observer):

    def __init__(self, output_directory: str) -> None:
        """ Write function values of the front into files.

        :param output_directory: Output directory. Each front will be saved on a file `FUN.x`. """
        self.counter = 0
        self.directory = output_directory

        if Path(self.directory).is_dir():
            LOGGER.warning('Directory {} exists. Removing contents.'.format(self.directory))
            for file in os.listdir(self.directory):
                os.remove('{0}/{1}'.format(self.directory, file))
        else:
            LOGGER.warning('Directory {} does not exist. Creating it.'.format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs['PROBLEM']
        solutions = kwargs['SOLUTIONS']

        if solutions:
            if isinstance(problem, DynamicProblem):
                termination_criterion_is_met = kwargs.get('TERMINATION_CRITERIA_IS_MET', None)

                if termination_criterion_is_met:
                    print_function_values_to_file(solutions, '{}/FUN.{}'.format(self.directory, self.counter))
                    self.counter += 1
            else:
                print_function_values_to_file(solutions, '{}/FUN.{}'.format(self.directory, self.counter))
                self.counter += 1


class PlotFrontToFileObserver(Observer):

    def __init__(self, output_directory: str) -> None:
        self.directory = output_directory
        self.plot_front = Plot(plot_title='Pareto front approximation')
        self.last_front = []
        self.fronts = []
        self.counter = 0

        if Path(self.directory).is_dir():
            LOGGER.warning('Directory {} exists. Removing contents.'.format(self.directory))
            for file in os.listdir(self.directory):
                os.remove('{0}/{1}'.format(self.directory, file))
        else:
            LOGGER.warning('Directory {} does not exist. Creating it.'.format(self.directory))
            Path(self.directory).mkdir(parents=True)

    def update(self, *args, **kwargs):
        problem = kwargs['PROBLEM']
        solutions = kwargs['SOLUTIONS']

        if solutions:
            if isinstance(problem, DynamicProblem):
                termination_criterion_is_met = kwargs.get('TERMINATION_CRITERIA_IS_MET', None)

                if termination_criterion_is_met:
                    if self.counter > 0:
                        igd = InvertedGenerationalDistance(self.last_front)
                        igd_value = igd.compute(solutions)
                    else:
                        igd_value = 1

                    if igd_value > 0.005:
                        self.fronts += solutions
                        self.plot_front.plot([self.fronts],
                                             label=[problem.get_name()],
                                             filename='{}/front-{}'.format(self.directory, self.counter))

                    self.counter += 1
                    self.last_front = solutions
            else:
                self.plot_front.plot([solutions], filename='{}/front-{}'.format(self.directory, self.counter))
                self.counter += 1


class VisualizerObserver(Observer):

    def __init__(self,
                 reference_front: List[S] = None,
                 reference_point: list = None,
                 display_frequency: float = 1.0) -> None:
        self.figure = None
        self.display_frequency = display_frequency

        self.reference_point = reference_point
        self.reference_front = reference_front

    def update(self, *args, **kwargs):
        evaluations = kwargs['EVALUATIONS']
        solutions = kwargs['SOLUTIONS']

        if solutions:
            if self.figure is None:
                self.figure = StreamingPlot(reference_point=self.reference_point,
                                            reference_front=self.reference_front)
                self.figure.plot(solutions)

            if (evaluations % self.display_frequency) == 0:
                # check if reference point has changed
                reference_point = kwargs.get('REFERENCE_POINT', None)

                if reference_point:
                    self.reference_point = reference_point
                    self.figure.update(solutions, reference_point)
                else:
                    self.figure.update(solutions)

                self.figure.ax.set_title('Eval: {}'.format(evaluations), fontsize=13)
