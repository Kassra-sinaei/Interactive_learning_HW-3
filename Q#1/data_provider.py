from amalearn.environment import EnvironmentBase
import gym
import numpy as np

class DataProviders(EnvironmentBase):
    def __init__(self, reward, episode_max_length, id, max_customer_number, income, penalty, Lambdas, gamma,
                 container=None):
        # Lambda is a list containing lambda values for company A & B respectively(2x2 list)
        state_space = gym.spaces.MultiDiscrete([21,21])
        action_space = gym.spaces.Discrete(11) # 0-> B buys vacancy from A
                                               # 5-> neither of companies buy vacancy
                                               # 10-> A buys 5 vacancies from B

        super(DataProviders, self).__init__(action_space, state_space, id, container)
        self.episode_max_length = episode_max_length
        #current state contains 2 integers which are number of companies costumers
        self.state = {
            'current_state': [10,10],
            'length': 0,
            'last_action': None
        }

        self.max_customer = max_customer_number     # Maximum number of costumers each company can handle
        self.income = income                        # Signing new contracts benefit each company by this amount
        self.penalty = penalty                      # Buying vacancy from another company costs this amount
        self.a_contract = reward(Lambdas[0][0])     # Poisson random number generator for new contracts @ A
        self.a_termination = reward(Lambdas[0][1])  # Poisson random number generator for terminated contracts @ A
        self.b_contract = reward(Lambdas[1][0])     # Poisson random number generator for new contracts @ B
        self.b_termination = reward(Lambdas[1][1])  # Poisson random number generator for terminated contracts @ A
        self.discount_factor = gamma
        self.V = np.zeros((self.max_customer + 1, self.max_customer + 1))

    def calculate_reward(self, action):
        ########################################################
        ###      This method takes care of calculating       ###
        ###     return of input action, before calling it    ###
        ###   remember to set state to desired state using   ###
        ###                 go_to_state(i,j)                 ###
        ########################################################
        reward = 0
        action -= 5  # 0-> B buys vacancy from A
                     # 5-> neither of companies buy vacancy
                     # 10-> A buys 5 vacancies from B
        # Penalize agent for duing invalid actions
        if self.state['current_state'][0] - action > 20 or self.state['current_state'][0] - action < 0\
                or self.state['current_state'][1] + action > 20 or self.state['current_state'][1] + action < 0:
            return -10000
        # New state after selling or Buying from another company
        new_state = [self.state['current_state'][0] - action, self.state['current_state'][1] + action]

        reward += -self.penalty * abs(action) # This value is always negative and is cost of selling or buying vacancy

        for a_new in range(self.a_contract.min,self.a_contract.max):
            for b_new in range(self.b_contract.min, self.b_contract.max):
                for a_terminated in range(self.a_termination.min,self.a_termination.max):
                    for b_terminated in range(self.b_termination.min,self.b_termination.max):

                        if new_state[0] - a_terminated < 0 or new_state[1] - b_terminated < 0:
                            continue

                        prob = self.a_contract.vals[a_new] * self.b_contract.vals[b_new] * \
                               self.a_termination.vals[a_terminated] * self.b_termination.vals[b_terminated]
                        # Check if Companies can accept more customers or not
                        if new_state[0] + a_new > 20:
                            reward -= 5 * (20 - a_new + next_state[0])
                        if new_state[1] + b_new > 20:
                            reward -= 5 * (20 - b_new + next_state[0])
                        # Rest of the calculations based on companies maximum capacity
                        max_contract_a = min(new_state[0], a_new)
                        max_contract_b = min(new_state[1],b_new)

                        r = (max_contract_a + max_contract_b) * self.income

                        #new state after signing and terminating contracts
                        next_state = [max(min(new_state[0] - max_contract_a + a_terminated, self.max_customer),0),
                                      max(min(new_state[1] - max_contract_b + b_terminated, self.max_customer), 0)]

                        reward += prob * (r + self.discount_factor * self.V[next_state[0]][next_state[1]])


        return reward

    # This method is implemented for switching between states during dynamic programing
    def go_to_state(self, a, b):
        self.state['current_state'][0] = a
        self.state['current_state'][1] = b

    def terminated(self):
        return self.state['length'] >= self.episode_max_length

    def observe(self):
        return self.state['current_state']

    def available_actions(self):
        return self.action_space.n

    def next_state(self,action):
        self.state['length'] += 0
        self.state['last_action'] = action

    def reset(self):
        self.state['length'] = 0
        self.state['last_action'] = None

    def render(self, mode='human'):
        print('{}:\taction={}'.format(self.state['length'], self.state['last_action']))

    def close(self):
        return