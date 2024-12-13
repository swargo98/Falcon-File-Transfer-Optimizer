import logging
import numpy as np
from skopt.space import Integer
from skopt import Optimizer
import time

# Simple black box function for testing
def black_box_function(params):
    """
    A simple black box function that evaluates a given set of parameters.
    In this case, it's just a quadratic function for simplicity.
    """
    # Example: return the sum of squares of the parameters, the optimizer will minimize this
    logging.info(f"Executing black box function with params: {params}")
    return sum([x**2 - 9*x + 20 for x in params])

def run_probe(current_cc, count, verbose, logger, black_box_function):
    if verbose:
        logger.info("Iteration {0} Starts ...".format(count))

    t1 = time.time()
    current_value = black_box_function([current_cc])
    t2 = time.time()

    if verbose:
        logger.info("Iteration {0} Ends, Took {1} Seconds. Score: {2}.".format(
            count, np.round(t2-t1, 2), current_value))

    return current_value

def gradient_opt_fast(configurations, black_box_function, logger, verbose=True):
    max_thread, count = configurations["thread_limit"], 0
    soft_limit, least_cost = max_thread, 0
    values = []
    ccs = [1]
    theta = 0

    while True:
        count += 1
        values.append(run_probe(ccs[-1], count, verbose, logger, black_box_function))

        if values[-1] == 10 ** 10:
            logger.info("Optimizer Exits ...")
            break

        if count == 20:
                break

        if values[-1] < least_cost:
            least_cost = values[-1]
            soft_limit = min(ccs[-1]+10, max_thread)

        if len(ccs) == 1:
            ccs.append(2)

        else:
            dist = max(1, np.abs(ccs[-1] - ccs[-2]))
            if ccs[-1]>ccs[-2]:
                gradient = (values[-1] - values[-2])/dist
            else:
                gradient = (values[-2] - values[-1])/dist

            if values[-2] !=0:
                gradient_change = np.abs(gradient/values[-2])
            else:
                gradient_change = np.abs(gradient)

            if gradient>0:
                if theta <= 0:
                    theta -= 1
                else:
                    theta = -1

            else:
                if theta >= 0:
                    theta += 1
                else:
                    theta = 1


            update_cc = int(theta * np.ceil(ccs[-1] * gradient_change))
            next_cc = min(max(ccs[-1] + update_cc, 2), soft_limit)
            # print("curr limit: ", least_cost, soft_limit)
            logger.info("Gradient: {0}, Gredient Change: {1}, Theta: {2}, Previous CC: {3}, Choosen CC: {4}".format(gradient, gradient_change, theta, ccs[-1], next_cc))
            ccs.append(next_cc)

    return [ccs[-1]]

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
    best_params = gradient_opt_fast(configurations, black_box_function, logger, verbose=True)
    logger.info(f"Optimization completed. Best parameters: {best_params}")

if __name__ == '__main__':
    main()
