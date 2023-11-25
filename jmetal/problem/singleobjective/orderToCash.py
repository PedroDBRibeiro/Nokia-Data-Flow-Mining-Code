from jmetal.core.problem import BinaryProblem
from jmetal.core.solution import BinarySolution
import random
import numpy
import math

from jmetal.util.observer import DimensionWiseDiversityObserver

class orderToCash(BinaryProblem):

    def __init__(self, milestones: list, full_datamodel_cols: list ,number_of_variables: int = 1, number_of_objectives: int = 1):
        
        super().__init__()

        self.full_datamodel_cols = full_datamodel_cols
        self.milestones = milestones
        
        self.number_of_variables = number_of_variables
        self.number_of_objectives = number_of_objectives
        self.number_of_bits = len(full_datamodel_cols)


        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['Count Ones']
        

        self.w1 = -1
        self.w2 = -10
        self.w5 = 0.001

        self.milestone_array = [[1 if j in i.columns else 0 for j in self.full_datamodel_cols] for i in self.milestones ]


    def contains_all_ones(self, milestones, solution):
        for i in range(len(milestones)):
            if milestones[i] == 1 and solution[i] != 1:
                return 0
        return 1

    def evaluate(self, solution: BinarySolution) -> BinarySolution:

        on_bits = solution.variables[0].count(1)
        coverage = on_bits / self.number_of_bits

        subset_cardinality = 0
        for i in self.milestone_array:
            subset_cardinality += self.contains_all_ones(i,solution.variables[0])
            
        complexity = math.pow(on_bits, 3/2) * math.log(on_bits)

        solution.objectives[0] = ((self.w1 * coverage) + (self.w2 * (subset_cardinality/len(self.milestones)))) / (self.w5 * complexity) 
        #solution.objectives[0] = ( (self.w1 * coverage) + (self.w2 * (subset_cardinality/len(self.milestones))) / (self.w5 * complexity) )


        return solution
    
    #self.xpl, self.xpt = self.diversity_calculator.update()
        #for i in self.milestones:
        #    #milestone_array = [1 if j in i.columns else 0 for j in self.full_datamodel_cols]
        #    subset_cardinality += self.contains_all_ones(milestone_array,solution.variables[0])
        #(subset_cardinality/len(self.milestones))
        #+ self.w3 * xpl + self.w4 * xpt)    

    def create_solution(self) -> BinarySolution:
        new_solution = BinarySolution(number_of_variables=self.number_of_variables,
                                      number_of_objectives=self.number_of_objectives)
        

        random_sample_milestones = random.sample(self.milestones, random.randint(1,len(self.milestones)))
        column_collection = []
        for milestones in random_sample_milestones:
            column_collection.append(milestones.columns)

        column_concat = numpy.concatenate([index.values for index in column_collection])

        if random.random() < 0.35:
            new_solution.variables[0] = \
                [1 if random.randint(0, 1) == 0 else 0 for _ in range(self.number_of_bits)]
        else:
            new_solution.variables[0] = \
                [1 if i in column_concat else 0 for i in self.full_datamodel_cols]

        return new_solution
    
    def get_name(self) -> str:
        return 'orderToCash'


if __name__ == '__main__':

    problem = orderToCash(10)
    problem.create_solution
    print(problem)
    