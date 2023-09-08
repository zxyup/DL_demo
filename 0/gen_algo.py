import numpy as np
import math

# Parameters
D_center = 110  # Depth at center
alpha = math.radians(1.5)  # Slope angle in radians
theta = math.radians(120)  # Transducer opening angle in radians
width = 4 * 1852  # East-West width in meters
d_min = 2 * D_center * math.tan(theta/2) * 0.10  # Minimum distance between two lines based on 10% overlap
d_max = 2 * D_center * math.tan(theta/2) * 0.20  # Maximum distance between two lines based on 20% overlap

# Genetic Algorithm parameters
population_size = 100
num_generations = 200
mutation_rate = 0.2
crossover_rate = 0.8

# Fitness function
def fitness(chromosome):
    coverage = 0
    last_line = 0
    for line in chromosome:
        coverage_width = 2 * (D_center + (line - width/2) * math.tan(alpha)) * math.tan(theta/2)
        overlap = coverage_width - (line - last_line)
        if overlap < d_min or overlap > d_max:
            return 0  # Invalid solution
        coverage += coverage_width
        last_line = line
    return coverage

def genetic_algorithm():
    # Initialize a population of potential solutions
    population = np.random.uniform(low=d_min, high=d_max, size=(population_size, int(width/d_min)))

    # Genetic Algorithm
    for generation in range(num_generations):
        # Evaluate fitness
        fitness_values = [fitness(chromo) for chromo in population]

        # Select parents
        parents = np.argsort(fitness_values)[-2:]

        # Crossover
        if np.random.rand() < crossover_rate:
            crossover_point = np.random.randint(1, population.shape[1]-1)
            child1 = np.concatenate((population[parents[0], :crossover_point], population[parents[1], crossover_point:]))
            child2 = np.concatenate((population[parents[1], :crossover_point], population[parents[0], crossover_point:]))
            population = np.vstack((population, child1, child2))

        # Mutation
        for chromo in population:
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(population.shape[1])
                chromo[mutation_point] += np.random.uniform(-d_min/10, d_min/10)

        # Select the next generation
        fitness_values = [fitness(chromo) for chromo in population]
        population = population[np.argsort(fitness_values)[-population_size:]]

    return population[np.argmax([fitness(chromo) for chromo in population])]

if __name__ == "__main__":
    best_solution = genetic_algorithm()
    print("最佳测线位置列表（东-西方向，单位为米）:", best_solution)
