from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver
from jmetal.operator.mutation import PolynomialMutation 
from jmetal.problem import DTLZ1
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == '__main__':
    problem = DTLZ1(number_of_objectives=5)
    max_evaluations = 100000

    algorithm = SMPSO(
            problem=problem,
            swarm_size=100,
            mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,distribution_index=20),
            leaders=CrowdingDistanceArchive(100),
            termination_criterion=StoppingByEvaluations(max=max_evaluations)
    )

    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print("Best solution:", result[0])
    print("Whose fitness is:", result[1])
    print('Computing time: ' + str(algorithm.total_computing_time))
