import pygmo as pg

from jmetal.core.problem import FloatProblem, S

'''
 No. Functions fi*=fi(x*)
Unimodal Functions
1 Sphere Function -1400
2 Rotated High Conditioned Elliptic Function -1300
3 Rotated Bent Cigar Function -1200
4 Rotated Discus Function -1100
5 Different Powers Function -1000

Basic Multimodal Functions
6 Rotated Rosenbrock’s Function -900
7 Rotated Schaffers F7 Function -800
8 Rotated Ackley’s Function -700
9 Rotated Weierstrass Function -600
10 Rotated Griewank’s Function -500
11 Rastrigin’s Function -400
12 Rotated Rastrigin’s Function -300
13 Non-Continuous Rotated Rastrigin’s Function -200
14 Schwefel's Function -100
15 Rotated Schwefel's Function 100
16 Rotated Katsuura Function 200
17 Lunacek Bi_Rastrigin Function 300
18 Rotated Lunacek Bi_Rastrigin Function 400
19 Expanded Griewank’s plus Rosenbrock’s Function 500
20 Expanded Scaffer’s F6 Function 600

Composition Functions
21 Composition Function 1 (n=5,Rotated) 700
22 Composition Function 2 (n=3,Unrotated) 800
23 Composition Function 3 (n=3,Rotated) 900
24 Composition Function 4 (n=3,Rotated) 1000
25 Composition Function 5 (n=3,Rotated) 1100
26 Composition Function 6 (n=5,Rotated) 1200
27 Composition Function 7 (n=5,Rotated) 1300
28 Composition Function 8 (n=5,Rotated) 1400
'''


class CEC2013(FloatProblem):

    def __init__(self, function_type: int = 0, number_of_variables: int = 0):
        """ jMetal common structure """
        super(CEC2013, self).__init__()
        self.number_of_objectives = 1
        self.number_of_constraints = 0

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Fitness']

        self.function_type = function_type
        self.number_of_variables = number_of_variables

        udp = pg.cec2013(prob_id=function_type, dim=number_of_variables)

        self.benchmark = pg.problem(udp)
        self.set_bounds()

    def evaluate(self, solution: S) -> S:
        solution.objectives[0] = self.benchmark.fitness(solution.variables)[0]

        return solution

    def get_name(self) -> str:
        return "CEC_2013"

    def set_bounds(self):
        self.lower_bound = self.benchmark.get_bounds()[0].tolist()
        self.upper_bound = self.benchmark.get_bounds()[1].tolist()
