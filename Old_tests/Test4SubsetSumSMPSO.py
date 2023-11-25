from jmetal.problem.multiobjective.unconstrained import SubsetSum
from jmetal.algorithm.multiobjective.smpso import SMPSO
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.util.observer import ProgressBarObserver
from jmetal.operator.mutation import PolynomialMutation 
from jmetal.util.termination_criterion import StoppingByEvaluations


from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm 
from jmetal.operator import BitFlipMutation
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.selection import BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations


if __name__ == '__main__':

    problem = SubsetSum(C = 1000 , W= [100 , 400, 300, 5000, 130, 200])

    max_evaluations = 100000

    #algorithm = SMPSO(
     #       problem=problem,
     #      swarm_size=100,
     #       mutation=PolynomialMutation(probability=1.0 / problem.number_of_variables,distribution_index=20),
     #      leaders=CrowdingDistanceArchive(100),
     #       termination_criterion=StoppingByEvaluations(max=max_evaluationsimage.png)
    # )

    algorithm = GeneticAlgorithm(
        problem = problem,
        population_size = 20,
        offspring_population_size = 20,
        mutation = BitFlipMutation(probability=1/problem.number_of_variables),
        crossover=  SPXCrossover(probability=0.85),
        selection = BestSolutionSelection(),
        termination_criterion =  StoppingByEvaluations(max= max_evaluations)
    )


    algorithm.observable.register(observer=ProgressBarObserver(max=max_evaluations))
    
    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + result.get_binary_string())
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))