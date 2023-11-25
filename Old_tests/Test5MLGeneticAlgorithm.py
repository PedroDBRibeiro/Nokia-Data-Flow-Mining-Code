from jmetal.problem.singleobjective.orderToCash import orderToCash


if __name__ == '__main__':
    #variables = [random.choice([0,1]) for i in range(10)]
    #print(variables)

    problem  = orderToCash(number_of_variables=10, number_of_objectives=2)
    solution = problem.create_solution()
    print(solution)



