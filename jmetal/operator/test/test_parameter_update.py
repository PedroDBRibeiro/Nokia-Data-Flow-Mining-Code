import unittest
import numpy.testing as npt
from jmetal.operator.parameter_update import DiversityBasedControl


class DiversityBasedControlTestCases(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        intervals = [(0, 1), (0, 1)]
        cls.diversity_control = DiversityBasedControl(intervals=intervals, max_evaluations=1000)

    def test_calculate_alpha_diversification(self):
        alpha = self.diversity_control.get_alpha_diversification(1)
        npt.assert_almost_equal(alpha, 0.999, decimal=3)

        alpha = self.diversity_control.get_alpha_diversification(2)
        npt.assert_almost_equal(alpha, 0.998, decimal=3)

        alpha = self.diversity_control.get_alpha_diversification(100)
        npt.assert_almost_equal(alpha, 0.900, decimal=3)

        alpha = self.diversity_control.get_alpha_diversification(250)
        npt.assert_almost_equal(alpha, 0.750, decimal=3)

        alpha = self.diversity_control.get_alpha_diversification(500)
        npt.assert_almost_equal(alpha, 0.500, decimal=3)

        alpha = self.diversity_control.get_alpha_diversification(750)
        npt.assert_almost_equal(alpha, 0.250, decimal=3)

        alpha = self.diversity_control.get_alpha_diversification(1000)
        npt.assert_almost_equal(alpha, 0.000, decimal=3)

    def test_calculate_alpha_intensification(self):
        alpha = self.diversity_control.get_alpha_intensification(1)
        npt.assert_almost_equal(alpha, 0.001, decimal=3)

        alpha = self.diversity_control.get_alpha_intensification(2)
        npt.assert_almost_equal(alpha, 0.002, decimal=3)

        alpha = self.diversity_control.get_alpha_intensification(100)
        npt.assert_almost_equal(alpha, 0.100, decimal=3)

        alpha = self.diversity_control.get_alpha_intensification(250)
        npt.assert_almost_equal(alpha, 0.250, decimal=3)

        alpha = self.diversity_control.get_alpha_intensification(500)
        npt.assert_almost_equal(alpha, 0.500, decimal=3)

        alpha = self.diversity_control.get_alpha_intensification(750)
        npt.assert_almost_equal(alpha, 0.750, decimal=3)

        alpha = self.diversity_control.get_alpha_intensification(1000)
        npt.assert_almost_equal(alpha, 1.000, decimal=3)

    def test_calculate_diversification_percentage(self):
        xpl = 0.9
        self.diversity_control.set_diversification_percentage(evaluation=1, xpl=xpl)
        percent = self.diversity_control.get_diversification_percentage()
        npt.assert_almost_equal(percent, 0.0999, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=2, xpl=xpl)
        percent = self.diversity_control.get_diversification_percentage()
        npt.assert_almost_equal(percent, 0.0998, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=100, xpl=xpl)
        percent = self.diversity_control.get_diversification_percentage()
        npt.assert_almost_equal(percent, 0.0900, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=250, xpl=xpl)
        percent = self.diversity_control.get_diversification_percentage()
        npt.assert_almost_equal(percent, 0.0750, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=500, xpl=xpl)
        percent = self.diversity_control.get_diversification_percentage()
        npt.assert_almost_equal(percent, 0.0500, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=750, xpl=xpl)
        percent = self.diversity_control.get_diversification_percentage()
        npt.assert_almost_equal(percent, 0.0250, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=1000, xpl=xpl)
        percent = self.diversity_control.get_diversification_percentage()
        npt.assert_almost_equal(percent, 0.0000, decimal=4)

    def test_calculate_intensification_percentage(self):
        xpt = 0.1
        self.diversity_control.set_intensification_percentage(evaluation=1, xpt=xpt)
        percent = self.diversity_control.get_intensification_percentage()
        npt.assert_almost_equal(percent, 0.0009, decimal=4)

        self.diversity_control.set_intensification_percentage(evaluation=2, xpt=xpt)
        percent = self.diversity_control.get_intensification_percentage()
        npt.assert_almost_equal(percent, 0.0018, decimal=4)

        self.diversity_control.set_intensification_percentage(evaluation=100, xpt=xpt)
        percent = self.diversity_control.get_intensification_percentage()
        npt.assert_almost_equal(percent, 0.0900, decimal=4)

        self.diversity_control.set_intensification_percentage(evaluation=250, xpt=xpt)
        percent = self.diversity_control.get_intensification_percentage()
        npt.assert_almost_equal(percent, 0.2250, decimal=4)

        self.diversity_control.set_intensification_percentage(evaluation=500, xpt=xpt)
        percent = self.diversity_control.get_intensification_percentage()
        npt.assert_almost_equal(percent, 0.4500, decimal=4)

        self.diversity_control.set_intensification_percentage(evaluation=750, xpt=xpt)
        percent = self.diversity_control.get_intensification_percentage()
        npt.assert_almost_equal(percent, 0.6750, decimal=4)

        self.diversity_control.set_intensification_percentage(evaluation=1000, xpt=xpt)
        percent = self.diversity_control.get_intensification_percentage()
        npt.assert_almost_equal(percent, 0.9000, decimal=4)

    def test_calculate_do_nothing_percentage(self):
        xpl = 0.9
        xpt = 0.1
        self.diversity_control.set_diversification_percentage(evaluation=1, xpl=xpl)
        self.diversity_control.set_intensification_percentage(evaluation=1, xpt=xpt)
        self.diversity_control.set_do_nothing_percentage()
        percent = self.diversity_control.get_do_nothing_percentage()
        npt.assert_almost_equal(percent, 0.8992, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=2, xpl=xpl)
        self.diversity_control.set_intensification_percentage(evaluation=2, xpt=xpt)
        self.diversity_control.set_do_nothing_percentage()
        percent = self.diversity_control.get_do_nothing_percentage()
        npt.assert_almost_equal(percent, 0.8984, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=100, xpl=xpl)
        self.diversity_control.set_intensification_percentage(evaluation=100, xpt=xpt)
        self.diversity_control.set_do_nothing_percentage()
        percent = self.diversity_control.get_do_nothing_percentage()
        npt.assert_almost_equal(percent, 0.8200, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=250, xpl=xpl)
        self.diversity_control.set_intensification_percentage(evaluation=250, xpt=xpt)
        self.diversity_control.set_do_nothing_percentage()
        percent = self.diversity_control.get_do_nothing_percentage()
        npt.assert_almost_equal(percent, 0.7000, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=500, xpl=xpl)
        self.diversity_control.set_intensification_percentage(evaluation=500, xpt=xpt)
        self.diversity_control.set_do_nothing_percentage()
        percent = self.diversity_control.get_do_nothing_percentage()
        npt.assert_almost_equal(percent, 0.500, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=750, xpl=xpl)
        self.diversity_control.set_intensification_percentage(evaluation=750, xpt=xpt)
        self.diversity_control.set_do_nothing_percentage()
        percent = self.diversity_control.get_do_nothing_percentage()
        npt.assert_almost_equal(percent, 0.3000, decimal=4)

        self.diversity_control.set_diversification_percentage(evaluation=1000, xpl=xpl)
        self.diversity_control.set_intensification_percentage(evaluation=1000, xpt=xpt)
        self.diversity_control.set_do_nothing_percentage()
        percent = self.diversity_control.get_do_nothing_percentage()
        npt.assert_almost_equal(percent, 0.1000, decimal=4)

    def test_calculate_all_percentages(self):
        update_info = {"EVALUATIONS": 1,
                       "XPL": 0.9,
                       "XPT": 0.1}

        arr_result = self.diversity_control.update(**update_info)
        npt.assert_array_almost_equal(arr_result, [0.8992, 0.0999, 0.0009], decimal=4)

        update_info = {"EVALUATIONS": 2,
                       "XPL": 0.9,
                       "XPT": 0.1}

        arr_result = self.diversity_control.update(**update_info)
        npt.assert_array_almost_equal(arr_result, [0.8984, 0.0998, 0.0018], decimal=4)

        update_info = {"EVALUATIONS": 100,
                       "XPL": 0.9,
                       "XPT": 0.1}

        arr_result = self.diversity_control.update(**update_info)
        npt.assert_array_almost_equal(arr_result, [0.8200, 0.0900, 0.0900], decimal=4)

        update_info = {"EVALUATIONS": 250,
                       "XPL": 0.9,
                       "XPT": 0.1}

        arr_result = self.diversity_control.update(**update_info)
        npt.assert_array_almost_equal(arr_result, [0.7000, 0.0750, 0.2250], decimal=4)

        update_info = {"EVALUATIONS": 500,
                       "XPL": 0.9,
                       "XPT": 0.1}

        arr_result = self.diversity_control.update(**update_info)
        npt.assert_array_almost_equal(arr_result, [0.5000, 0.0500, 0.4500], decimal=4)

        update_info = {"EVALUATIONS": 750,
                       "XPL": 0.9,
                       "XPT": 0.1}

        arr_result = self.diversity_control.update(**update_info)
        npt.assert_array_almost_equal(arr_result, [0.3000, 0.0250, 0.6750], decimal=4)

        update_info = {"EVALUATIONS": 1000,
                       "XPL": 0.9,
                       "XPT": 0.1}

        arr_result = self.diversity_control.update(**update_info)
        npt.assert_array_almost_equal(arr_result, [0.1000, 0.0000, 0.9000], decimal=4)


if __name__ == '__main__':
    unittest.main()
