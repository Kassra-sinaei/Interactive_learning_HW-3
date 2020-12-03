from agent import Agent
from data_provider import DataProviders
from poisson import Poisson
import matplotlib.pyplot as plt
import time

def train(agent):
    evaluate_percision = 10
    while True:
        agent.policy_eval(theta = evaluate_percision)
        evaluate_percision /= 2
        print("policy evaluated")
        stable = agent.policy_iter()
        print("policy improved")
        if stable:
            break

    print("State Values:\n", agent.environment.V)
    print("Optimal Policy:\n", agent.policy)

    plt.imshow(agent.V, cmap='hot', interpolation='nearest')
    plt.show()
    plt.imshow(agent.policy, cmap='hot', interpolation='nearest')
    plt.show()

if __name__ == "__main__":
    env = DataProviders(reward = Poisson, episode_max_length = 20, id = '1',
                            max_customer_number = 20, income = 10, penalty = 2,gamma = 0.9, Lambdas = list([[3,3],[4,2]]))
    agent = Agent(id = '1', environment = env)
    print('hi')
    start = time.time()
    train(agent)
    print((time.time() - start)/60)
    pass
