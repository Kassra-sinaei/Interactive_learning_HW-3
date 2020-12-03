from scipy.stats import poisson

########################################################
###     This class is defined to generate number     ###
###       of contracts signed or ended               ###
########################################################
class Poisson:
    def __init__(self, Lambda):

        self.min = 0
        state = True
        self.vals = {}
        sum = 0

        while True:
            if state:
                temp = poisson.pmf(self.min, Lambda)
                if temp <= 0.01:
                    self.min += 1
                else:
                    self.vals[self.min] = temp
                    sum += temp
                    self.max = self.min + 1
                    state = False
            elif state == False:
                temp = poisson.pmf(self.max, Lambda)
                if temp > 0.01:
                    self.vals[self.max] = temp
                    sum += temp
                    self.max += 1
                else:
                    break

        # Normalizing pmf
        # values of n outside (self.min , self.max) has been set to zero
        added_val = (1 - sum) / (self.max - self.min)
        for key in self.vals:
            self.vals[key] += added_val


    def get_reward(self,n):
        # This method returns all the values in poisson distribution
        # with probability more than 0.01
        return self.vals[n]

if __name__ == "__main__":
    p = Poisson(4)
    print(p.vals)
    print(p.min,p.max)