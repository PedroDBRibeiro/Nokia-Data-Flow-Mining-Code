from abc import abstractmethod

import numpy as np
import angles


class Correction:
    @abstractmethod
    def correct_boundaries(self, upper_bound, lower_bound, value, original_value):
        pass


class RandomCorrection(Correction):

    def correct_boundaries(self, upper_bound, lower_bound, value, original_value=None):
        if value > upper_bound or value < lower_bound:
            value = np.random.uniform(lower_bound, upper_bound)

        return value


'''
Truncates the value accordingly to the boundary configuration
'''


class LimitCorrection(Correction):

    def correct_boundaries(self, upper_bound, lower_bound, value, original_value):
        return np.clip(value, lower_bound, upper_bound)


'''
Creates the correction as proposed in “JADE: Adaptive differential evolution with optional 
external archive" - J. Zhang and A. C. Sanderson.
'''


class JadeCorrection(Correction):
    def correct_boundaries(self, upper_bound, lower_bound, value, original_value):
        if value < lower_bound:
            value = (lower_bound + original_value) / 2
        elif value > upper_bound:
            value = (upper_bound + original_value) / 2
        return value


'''
Angle Correction -> Ciclical
'''


class AngleCorrection(Correction):

    def correct_boundaries(self, upper_bound, lower_bound, value, original_value):
        if value > upper_bound or value < lower_bound:
            value = float(format(angles.normalize(value, -180, 180), '.4f'))

        return value
