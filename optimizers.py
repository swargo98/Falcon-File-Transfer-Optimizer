from skopt.space import Integer
from skopt import Optimizer, dummy_minimize
from scipy.optimize import minimize
import numpy as np
import time
import logging

from abc import ABC, abstractmethod

class OptimizerBase(ABC):
    def __init__(self, configurations, logger, verbose=True):
        self.configurations = configurations
        self.logger = logger
        self.verbose = verbose
        self.max_thread_count = configurations.get("thread_limit", 1)
    
    def black_box_function(self, params):
        """
        The common black box function for all optimizers.
        This can be overridden by any child class if needed.
        """
        self.logger.info(f"Executing black box function with params: {params}")
        result = sum([x**2 - 9*x + 20 for x in params])  # Just an example, replace with your actual function logic
        return result
    
    def run_probe(self, concurrency, iteration_count):
        if self.verbose:
            self.logger.info("Iteration {0} Starts ...".format(iteration_count))

        t1 = time.time()
        current_value = self.black_box_function([concurrency])
        t2 = time.time()

        if self.verbose:
            self.logger.info("Iteration {0} Ends, Took {1} Seconds. Score: {2}.".format(
                iteration_count, np.round(t2-t1, 2), current_value))

        return current_value
    
    
    @abstractmethod
    def optimize(self):
        """The method that must be implemented by all child classes."""
        pass

class BayesOptimizer(OptimizerBase):
    def optimize(self):
        observation_limit, count = 25, 0
        max_thread_count_local = self.max_thread_count
        iterations = self.configurations.get("bayes", {}).get("num_of_exp", 5)
        mp_opt = self.configurations.get("mp_opt", False)

        if mp_opt:
            search_space  = [
                Integer(1, max_thread_count_local),  # Concurrency
                Integer(1, 10),          # Parallelism
                Integer(1, 10),          # Pipeline
                Integer(5, 20),          # Chunk/Block Size in KB: power of 2
            ]
        else:
            search_space = [
                Integer(1, max_thread_count_local),  # Concurrency
            ]

        params = []
        optimizer = Optimizer(
            dimensions=search_space,
            base_estimator="GP",  # Gaussian Process (GP)
            acq_func="gp_hedge",  # Acquisition function (gp_hedge)
            acq_optimizer="auto",  # Acquisition optimizer (auto)
            n_random_starts=self.configurations.get("bayes", {}).get("initial_run", 3),
            model_queue_size=observation_limit,
        )

        while True:
            count += 1

            if len(optimizer.yi) > observation_limit:
                optimizer.yi = optimizer.yi[-observation_limit:]
                optimizer.Xi = optimizer.Xi[-observation_limit:]

            if self.verbose:
                self.logger.info(f"Iteration {count} Starts...")

            t1 = time.time()
            res = optimizer.run(func=self.black_box_function, n_iter=1)
            t2 = time.time()

            if self.verbose:
                self.logger.info(f"Iteration {count} Ends, Took {np.round(t2-t1, 2)} Seconds. Best Params: {res.x} and Score: {res.fun}.")

            last_utility_value = optimizer.yi[-1] # Utility value, we have to minimize this
            last_concurrency = optimizer.Xi[-1][0]
            
            if last_utility_value == self.configurations.get("exit_value", 10 ** 10):
                self.logger.info("Optimizer Exits ...")
                break
            
            # Update the max_thread_count_local based on the last utility value if iteration count is infinity
            if iterations < 1:
                reset = False
                # Narrow down the search space if the utility value is positive
                if (last_utility_value > 0) and (last_concurrency < max_thread_count_local):
                    max_thread_count_local = max(last_concurrency, 2)
                    reset = True
                # Widen the search space if the utility value is negative and there is still room to increase concurrency
                if (last_utility_value < 0) and (last_concurrency == max_thread_count_local) and (last_concurrency < self.max_thread_count_local):
                    max_thread_count_local = min(last_concurrency + 5, self.max_thread_count)
                    reset = True

                if reset:
                    search_space[0] = Integer(1, max_thread_count_local)
                    optimizer = Optimizer(
                        dimensions=search_space,
                        n_initial_points=self.configurations.get("bayes", {}).get("initial_run", 3),
                        acq_optimizer="lbfgs",
                        model_queue_size=observation_limit,
                    )

            if iterations == count:
                self.logger.info(f"Best parameters: {res.x} and score: {res.fun}")
                params = res.x
                break

        return params
                
class GradientDescentOptimizer(OptimizerBase):
    def optimize(self):
        max_thread_count_local, count, min_cost = self.max_thread_count, 0, 0
        utility_values = []
        concurrencies = [1]
        direction = 0

        while True:
            count += 1
            utility_values.append(self.run_probe(concurrencies[-1], count))

            if utility_values[-1] == self.configurations.get("exit_value", 10 ** 10):
                self.logger.info("Optimizer Exits ...")
                break

            # remove the break statement after finalizing the code
            if count == 10:
                break

            if utility_values[-1] < min_cost:
                min_cost = utility_values[-1]
                max_thread_count_local = min(concurrencies[-1] + 10, self.max_thread_count)

            if len(concurrencies) == 1:
                concurrencies.append(2)

            else:
                dist = concurrencies[-1] - concurrencies[-2]
                gradient = (utility_values[-1] - utility_values[-2]) / (1 if dist==0 else dist)
                gradient_change = np.abs(gradient/(1 if utility_values[-2]==0 else utility_values[-2]))

                if gradient > 0:
                    direction = min(-1, direction - 1)
                else:
                    direction = max(1, direction + 1)

                update = int(direction * np.ceil(concurrencies[-1] * gradient_change))
                next_concurrency = min(max(2, concurrencies[-1] + update), max_thread_count_local)
                concurrencies.append(next_concurrency)
                self.logger.info(f"Gradient: {gradient}, Gredient Change: {gradient_change}, Theta: {direction}, Next Concurrency: {next_concurrency}.")

        return concurrencies[-1]
    
class HillClimbingOptimizer(OptimizerBase):
    def optimize(self):
        concurrencies = [1]
        direction, count = 1, 0
        current_utility, previous_utility = 0, -int(1e10) # Initialize with a large negative value as we are maximizing the utility

        while True:
            count += 1
            
            if self.verbose:
                self.logger.info(f"Iteration {count} Starts...")

            t1 = time.time()
            current_utility = self.black_box_function([concurrencies[-1]]) * -1 # Maximize the utility
            t2 = time.time()

            if self.verbose:
                self.logger.info(f"Iteration {count} Ends, Took {np.round(t2-t1, 2)} Seconds. Param: {[concurrencies[-1]]} Score: {current_utility}.")

            if current_utility == self.configurations.get("exit_value", 10 ** 10):
                self.logger.info("Optimizer Exits ...")
                break

            # remove the break statement after finalizing the code
            if count == 10:
                break

            if direction == 1:
                if current_utility > previous_utility:
                    concurrencies.append(min(concurrencies[-1] + 1, self.max_thread_count))
                    previous_utility = current_utility
                else:
                    concurrencies.append(max(1, concurrencies[-1] - 1))
                    # why no update of previous_utility here?
                    direction = 0

            elif direction == -1:
                if current_utility > previous_utility:
                    concurrencies.append(min(concurrencies[-1] + 1, self.max_thread_count))
                    # why no update of previous_utility here?
                    direction = 0
                else:
                    concurrencies.append(max(1, concurrencies[-1] - 1))
                    previous_utility = current_utility

            # No update would have workled here if it was if instead of elif
            elif direction == 0:
                change = (current_utility - previous_utility) / (previous_utility + 1e-10)
                previous_utility = current_utility
                if change > 0.1:
                    direction = 1
                    concurrencies.append(min(concurrencies[-1] + 1, self.max_thread_count))
                elif change < -0.1:
                    direction = -1
                    concurrencies.append(max(1, concurrencies[-1] - 1))
                else:
                    direction = 0

        return concurrencies[-1]
    
class BinarySearchOptimizer(OptimizerBase):
    def optimize(self):
        low, high = 1, self.max_thread_count
        count = 0
        utility_values = {}
        concurrencies = [low]

        while low <= high:
            count += 1

            utility_values[concurrencies[-1]] = self.run_probe(concurrencies[-1], count) * -1  # Maximize the utility

            if utility_values[concurrencies[-1]] == self.configurations.get("exit_value", 10 ** 10):
                self.logger.info("Optimizer Exits ...")
                break

            # remove the break statement after finalizing the code
            if count == 10:
                break

            if len(concurrencies) == 1:
                concurrencies.append(high)
                continue

            if len(concurrencies) == 2:
                mid = (low + high) // 2
                concurrencies.append(mid)
                continue

            self.logger.info(f"l, m, r: {low}:{utility_values[low]}, {mid}:{utility_values[mid]}, {high}:{utility_values[high]}")

            if utility_values[mid] > utility_values[high]:
                high = mid
            else:
                low = mid

            mid = (low + high) // 2
            concurrencies.append(mid)

        return [mid]
    
class BruteForceOptimizer(OptimizerBase):
    def optimize(self):
        utility_values = []

        for concurrency in range(1, self.max_thread_count+1):
            utility_values.append(self.black_box_function([concurrency]))

            if utility_values[-1] == self.configurations.get("exit_value", 10 ** 10):
                self.logger.info("Optimizer Exits ...")
                break

            # remove the break statement after finalizing the code
            if concurrency == 10:
                break

        best_concurrency = np.argmin(utility_values) + 1
        self.logger.info(f"Best concurrency: {best_concurrency} with utility value: {utility_values[best_concurrency-1]}")
        return [best_concurrency]
    
class CGOptimizer(OptimizerBase):
    def optimize(self):
        mp_opt = self.configurations["mp_opt"]

        if mp_opt:
            starting_params = [1, 1, 1, 10]
        else:
            starting_params = [1]

        optimizer = minimize(
            method="CG",
            fun=self.black_box_function,
            x0=starting_params,
            options= {
                "eps":1, # step size
            },
        )

        return optimizer.x


# Main function to run the optimizer
def main():
    # Example configurations for the optimizer
    configurations = {
        "thread_limit": 10,  # Max concurrency
        "bayes": {
            "num_of_exp": 5,  # Number of experiments
            "initial_run": 3  # Initial random starts
        },
        "mp_opt": False  # No multi-processing optimization
    }

    # Setting up the logger
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=log_format, level=logging.INFO)
    logger = logging.getLogger()

    # Run the optimizer with the black box function
    optimizer = CGOptimizer(configurations, logger, verbose=True)
    best_params = optimizer.optimize()
    logger.info(f"Optimization completed. Best parameters: {best_params}")

if __name__ == '__main__':
    main()
