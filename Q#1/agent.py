from amalearn.agent import AgentBase
import numpy as np


class Agent(AgentBase):
    def __init__(self, id, environment, discount_factor= 1.0):
        super(Agent, self).__init__(id, environment)

        self.discount_factor = discount_factor

        # Value and Policy for each state and each action
        self.V= np.zeros((self.environment.max_customer+1, self.environment.max_customer+1))
        self.policy = np.zeros((self.environment.max_customer+1, self.environment.max_customer+1),dtype=np.int)# Deterministic Policy

    def take_action(self) -> (object, float, bool, object):
        #return obs, r, d, i
        return {},0,False,{}

    def policy_eval(self,theta=0.5):
        print('evaluating policy...')
        while True:
            delta = 0
            # Loop on each state
            for i in range(self.V.shape[0]):
                for j in range(self.V.shape[1]):
                    old_value = self.V[i][j]
                    self.environment.go_to_state(i, j)
                    obs, Reward, d, x = self.environment.step(self.policy[i][j])
                    act_reward, contract_reward, probabilities, next_a, next_b = Reward
                    temp = act_reward
                    for index in range(len(probabilities)):
                        temp += probabilities[index] * (contract_reward[index] +
                                                         self.discount_factor * self.V[next_a[index]][next_b[index]])
                    self.V[i][j] = temp
                    delta = max(delta, abs(old_value-self.V[i][j]))
            if delta < theta:
                break

        pass

    def policy_iter(self):
        policy_stable = True
        print('improving policy...')
        # Loop on each state
        for i in range(self.V.shape[0]):
            for j in range(self.V.shape[1]):
                old_action = self.policy[i][j]

                max_act_val = None
                max_act = None

                a2b = min (i,5) + 5
                b2a = -min (j,5) + 5

                for action in range(b2a,a2b+1):
                    self.environment.go_to_state(i, j)
                    obs, Reward, d, x = self.environment.step(action)
                    act_reward, contract_reward, probabilities, next_a, next_b = Reward
                    sigma = act_reward
                    for index in range(len(probabilities)):
                        sigma += probabilities[index] * (contract_reward[index] +
                                                         self.discount_factor * self.V[next_a[index]][next_b[index]])
                    if max_act_val == None:
                        max_act_val = sigma
                        max_act = action
                    elif max_act_val < sigma:
                        max_act_val = sigma
                        max_act = action

                self.policy[i][j] = max_act

                if self.policy[i][j] != old_action:
                    policy_stable = False

        return policy_stable

    def value_iter(self):
        pass