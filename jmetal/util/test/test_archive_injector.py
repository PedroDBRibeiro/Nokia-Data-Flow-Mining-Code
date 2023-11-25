import unittest
import numpy as np
from jmetal.util.solutions.generator import ArchiveInjector


class ArchiveInjectorTest(unittest.TestCase):

    def setUp(self) -> None:
        self.file_path = "/home/phnarloch/Downloads/run-1.txt"
        self.generator = ArchiveInjector(self.file_path)

    def test_population_injection(self):
        result_list = self.generator.new()
        np.testing.assert_array_equal(result_list, [[180, 180, 180], [170, 170, 170], [160, 160, 160]])


if __name__ == '__main__':
    unittest.main()
