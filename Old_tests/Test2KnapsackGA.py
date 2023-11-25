from jmetal.problem.singleobjective.knapsack import Knapsack
from jmetal.algorithm.singleobjective.genetic_algorithm import GeneticAlgorithm 
from jmetal.operator import BitFlipMutation
from jmetal.operator.crossover import SPXCrossover
from jmetal.operator.selection import BestSolutionSelection
from jmetal.util.termination_criterion import StoppingByEvaluations

if __name__ == "__main__":

    problem = Knapsack(number_of_items = 5,
    capacity = 10,
    weights = [2,2,6,5,4],
    profits = [6,3,5,4,6],)

    max_evaluations = 20000

    algorithm = GeneticAlgorithm(
        problem = problem,
        population_size = 20,
        offspring_population_size = 20,
        mutation = BitFlipMutation(probability=1/problem.number_of_variables),
        crossover=  SPXCrossover(probability=0.85),
        selection = BestSolutionSelection(),
        termination_criterion =  StoppingByEvaluations(max= max_evaluations)
    )

    # run and get results
    algorithm.run()
    result = algorithm.get_result()

    print('Algorithm: ' + algorithm.get_name())
    print('Problem: ' + problem.get_name())
    print('Solution: ' + result.get_binary_string())
    print('Fitness:  ' + str(result.objectives[0]))
    print('Computing time: ' + str(algorithm.total_computing_time))


