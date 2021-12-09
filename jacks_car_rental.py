import numpy as np
import matplotlib.pyplot as plt
from itertools import product

class location():
    '''
    A car rental location

    Parameters
    ----------
    max_cars : int
        the maximum number of cars that can be in the location
    renting_reward_per_car : float
        the reward per car rented from the location
    requests_dist : (lambda function) requests_dist: (number_of_requests) -> float
        returns the probability of the car requests in a day to be number_of_requests 
        (we assume that the distribution of the number of requests in a day at the location is the same for any day and indepented of any other variable, such as day of the week etc.)
    returns_dist : (lambda function) returnss_dist: (number_of_returns) -> float
        returns the probability of the car returns in a day to be number_of_returns
        (we assume that the distribution of the number of returnss in a day at the location is the same for any day and indepented of any other variable, such as day of the week etc.)
        
    Attributes
    ----------
    max_cars : int
        the maximum number of cars that can be in the location
    renting_reward_per_car : float
        the reward per car rented from the location
    requests_dist : (lambda function) requests_dist: (number_of_requests) -> float
        returns the probability of the car requests in a day to be number_of_requests
        (we assume that the distribution of the number of requests in a day at the location is the same for any day and indepented of any other variable, such as day of the week etc.)
    returns_dist : (lambda function) returnss_dist: (number_of_returns) -> float
        returns the probability of the car returns in a day to be number_of_returns
        (we assume that the distribution of the number of returnss in a day at the location is the same for any day and indepented of any other variable, such as day of the week etc.)
    '''
    def __init__(self, max_cars, renting_reward_per_car, requests_dist, returns_dist):
        self.max_cars = max_cars
        self.renting_reward_per_car = renting_reward_per_car
        self.requests_dist = requests_dist
        self.returns_dist = returns_dist

class state():
    '''
    A state of the problem

    Parameters
    ----------
    cars : array_like (1d)
        the number of cars at each location, cars[i] -> number of cars in location i
    max_moving_cars_per_location : int
        the maximum number of cars that can be moved from each location (same for each location)

    Attributes
    ----------
    cars : array_like (1d)
        the number of cars at each location, cars[i] -> number of cars in location i
    max_moving_cars_per_location : int
        the maximum number of cars that can be moved from each location (same for each location)
    value : float
        the value of the state
    is_terminal : bool
        indicates if the state is a terminal state, if True then the state is a terminal state, if False than the state is not a terminal state
    actions : list 
        contains all the actions that can be taken at the state. Each action is a strictly lower triangular 2d matrix,
        action[i][j] -> cars to be moved from location i to location j (negative value means cars moving from location j to location i)
    '''
    def __init__(self, cars, max_moving_cars_per_location):
        self.cars = cars
        self.max_moving_cars_per_location = max_moving_cars_per_location
        temp = self.initial_evaluation()
        self.value = temp[0]
        self.is_terminal = temp[1]      
        self.actions = self.create_actions()

    def initial_evaluation(self):
        value = 0
        is_terminal = False
        for location_cars in self.cars:
            if location_cars == 0:
                value = -99999999999999999999999
                is_terminal = True
        return(value, is_terminal)

    def create_actions(self):
        available_actions = list()
        if self.is_terminal:
            available_actions.append(np.zeros((len(self.cars), len(self.cars))))
        else:
            actions = list()
            for i in range(len(self.cars) - 1):
                for j in range(i + 1, len(self.cars)):
                    actions.append(range(-min(self.cars[j], self.max_moving_cars_per_location), min(self.cars[i], self.max_moving_cars_per_location) + 1))
                    
            actions_flat = list(product(*actions))
            for action_flat in actions_flat:
                action = np.zeros((len(self.cars), len(self.cars)))
                k = 0
                for i in range(len(self.cars) - 1):
                    for j in range(i + 1, len(self.cars)):
                        action[i][j] = action_flat[k]
                        k += 1
                available_actions.append(action)
        return available_actions


class dynamics():
    '''
    Represents the problem itself

    Parameters
    ----------
    locations : array_like (1d)
        contains location objects, the locations of the problem
    max_moving_cars_per_location : int
        the maximum number of cars that can be moved from each location (same for each location)
    relocation_cost_per_car : 2d array_like
        a strictly lower triangular 2d matrix, relocation_cost_per_car[i][j] -> the cost of moving a car from location i to location j and vice versa (we assume that moving a car from location i to location j costs the same as moving a car from location j to location i, hence strictly lower triangular matrix)
    gamma : float 
        the discount rate
    theta : float
        the accuracy we want when evaluating the states under a policy

    Attributes
    ----------
    locations : array_like (1d)
        contains location objects, the locations of the problem
    max_moving_cars_per_location : int
        the maximum number of cars that can be moved from each location (same for each location)
    relocation_cost_per_car : 2d array_like
        a strictly lower triangular 2d matrix, relocation_cost_per_car[i][j] -> the cost of moving a car from location i to location j and vice versa (we assume that moving a car from location i to location j costs the same as moving a car from location j to location i, hence strictly lower triangular matrix)
    gamma : float 
        the discount rate
    theta : float
        the accuracy we want when evaluating the states under a policy
    states : list
        a list of all possible states of the problem, contains state objects
    policy : ndarray
        1d ndarray with same length as states. policy[i] -> the index of the action we take in the i-th state (action we take = self.states[i].actions[self.policy[i]])
    next_state_reward_prob : dictionary
        each element is a dictionary itself.
        contains the p(next_state, reward| current_state, action) for all possible (current_state, action), (next_state, reward) combinations,  
        next_state_reward_prob[current_state_index, action][next_state_index, reward] - > p(next_state, reward| current_state, action)       
    '''
    def __init__(self, locations, max_moving_cars_per_location, relocation_cost_per_car, gamma, theta):
        self.locations = locations
        self.max_moving_cars_per_location = max_moving_cars_per_location
        self.relocation_cost_per_car = relocation_cost_per_car 
        self.gamma = gamma
        self.theta = theta
        self.states = self.create_states()
        self.policy = np.zeros(len(self.states), dtype=int) 
        self.next_state_reward_prob = self.create_next_state_reward_prob()
    
    def create_states(self):
        #creates all the possible states of the problem, returns a list of state objects
        states = list()
        args = list()
        for location in self.locations:
            args.append(range(location.max_cars + 1))
        temp_states = list(product(*args))
        for temp_state in temp_states:           
            states.append(state(list(temp_state), self.max_moving_cars_per_location))  
        return states
        
    def return_request_probability(self, car_return, car_request):
        #given the car returns and the car requests at the locations (car_request[i] -> car returns at location i, car_request[i] -> car requests at location i) 
        #returns the probability of car_return and car_request happening (joint probability)
        probability = 1
        for i in range(len(self.locations)):
            if probability > 0:
                probability *= self.locations[i].requests_dist(car_request[i]) * self.locations[i].returns_dist(car_return[i])
        return probability

    def cars_moved_to_location(self, location_index, action):
        cars = 0
        for i in range(location_index + 1, len(self.locations)):
            cars -= action[location_index][i] #subtract the cars that move from the location to other locations
        for i in range(0, location_index):
            cars += action[i][location_index] #add the cars tha move into the location
        return cars
    
    def find_next_state_index(self, current_state, action, car_return, car_request):
        next_state_cars = np.zeros(len(self.locations), dtype=int)
        dims = np.zeros(len(self.locations), dtype=int)
        for i in range(len(self.locations)):
            next_state_cars[i] = current_state.cars[i] + car_return[i] - car_request[i] + self.cars_moved_to_location(i, action)  
            if next_state_cars[i] < 0:
                next_state_cars[i] = 0
            elif next_state_cars[i] > self.locations[i].max_cars:
                next_state_cars[i] = self.locations[i].max_cars
            dims[i] = self.locations[i].max_cars + 1
        next_state_index = np.ravel_multi_index(tuple(next_state_cars), tuple(dims))
        return next_state_index

    def compute_reward(self, car_request, action):
        #given the car requests at the locations (car_request[i] -> car requests at location i) and an action compute the overall reward
        reward = 0
        for i in range(len(self.locations)):
            reward += car_request[i] * self.locations[i].renting_reward_per_car
        for i in range(len(self.locations) - 1):
            for j in range(i + 1, len(self.locations)):
                reward -= abs(action[i][j]) * self.relocation_cost_per_car[i][j]
        return reward

    def next_state_reward_probability(self, state_index, action):
        #given the current state index and the action we take in that state returns a dictionary containing the probabilities of all possible
        #(next state index, reward) pairs. 
        args = list()
        for location in self.locations:
            args.append(range(location.max_cars + 1)) #the probability of requests=max_cars is already almost zero, we don't need to iterate further
        temp = list(product(*args))
        car_requests = temp
        car_returns = temp

        reward_next_state_prob = dict()
        for car_return in car_returns:
            for car_request in car_requests: 
                probability = self.return_request_probability(car_return, car_request)
                reward = self.compute_reward(car_request, action)
                next_state_index = self.find_next_state_index(self.states[state_index], action, car_return, car_request)
                if (next_state_index, reward) in reward_next_state_prob.keys():
                    reward_next_state_prob[(next_state_index, reward)] += probability
                else:
                    reward_next_state_prob[(next_state_index, reward)] = probability 
        return reward_next_state_prob
    
    def create_next_state_reward_prob(self):
        #returns a dictionary containing all possible (state_index, action) (next_state_index, reward) probabilities
        next_state_reward_prob = dict()
        for i in range(len(self.states)):
            for j in range(len(self.states[i].actions)):
                next_state_reward_prob[(i, j)] = self.next_state_reward_probability(i, self.states[i].actions[j])
        return next_state_reward_prob
        
    def compute_state_value(self, state_index, action_index):
        new_value = 0       
        for item in self.next_state_reward_prob[(state_index, action_index)].items():         
            new_value += item[1] * (item[0][1] + self.gamma * self.states[item[0][0]].value) 
        return new_value

    def policy_evaluation(self):
        '''
        evaluates the states under policy = self.policy, with accuracy self.theta
        '''
        delta = 2 * self.theta
        while delta >= self.theta:
            delta = 0
            for i in range(0, len(self.states)): 
                if not self.states[i].is_terminal:
                    old_value = self.states[i].value            
                    action_index = self.policy[i]           
                    self.states[i].value = self.compute_state_value(i, action_index) 
                    delta = max(delta, abs(old_value - self.states[i].value))   

    def policy_improvement(self):
        '''
        improves the current policy.

        Returns
        -------
        policy_stable : bool
            indicates if the policy changed or not. If it didn't, policy_stable is True, if it did policy_stable is False
        '''
        policy_stable = True
        for i in range(0, len(self.states)):
            if not self.states[i].is_terminal:
                old_action_index = self.policy[i]
                max = self.compute_state_value(i, 0)
                best_action_index = 0
                for j in range(1, len(self.states[i].actions)):
                    value = self.compute_state_value(i, j)
                    if value > max:
                        max = value
                        best_action_index = j         
                self.policy[i] = best_action_index
                if old_action_index != best_action_index:
                    policy_stable = False
        return policy_stable

    def plot_policy(self, policy_id):
        '''
        if the number of locations is 2, plots the policy and the state values.
        If there are more than 2 locations, prints the results
        '''
        if len(self.locations) == 2:
            fig_1 = plt.figure()
            ax_1 = fig_1.add_subplot()
            fig_2 = plt.figure()
            ax_2 = fig_2.add_subplot(projection='3d')

            x = np.zeros(len(self.states))
            y = np.zeros(len(self.states))
            z_1 = np.zeros(len(self.states))
            z_2 = np.zeros(len(self.states))
            policy_matrix = np.zeros((self.locations[0].max_cars + 1, self.locations[0].max_cars + 1))
            for i in range(0, len(self.states)):
                x[i] = self.states[i].cars[0]
                k = int(x[i])
                y[i] = self.states[i].cars[1]
                m = int(y[i])
                z_1[i] = self.states[i].actions[self.policy[i]][0][1]
                policy_matrix[k][m] = z_1[i]
                z_2[i] = self.states[i].value
                ax_1.text(x=x[i], y=y[i], s=z_1[i], va='center', ha='center')
            ax_1.matshow(policy_matrix, cmap=plt.cm.Pastel1)

            surf = ax_2.plot_trisurf(x, y, z_2)   
            surf._facecolors2d = surf._facecolor3d
            surf._edgecolors2d = surf._edgecolor3d
            
            ax_1.set(xlabel="cars at 1st location", ylabel="cars at 2nd location", title="action")    
            ax_2.set(xlabel="cars at 1st location", ylabel="cars at 2nd location", zlabel="state value")         

            fig_1.savefig("policy_" + str(policy_id) + ".png")
            fig_2.savefig("states_values" + str(policy_id) + ".png")
            plt.show()
            plt.close()
        else:
            for i in range(len(self.states)):
                print("state: " + str(self.states[i].cars) + ", action: " + str(self.states[i].actions[self.policy[i]]) + ", state value: " + str(self.states[i].value))

    def policy_iteration(self):
        policy_stable = False
        policy_id = 0
        while policy_stable == False:
            print("policy " + str(policy_id))
            self.policy_evaluation() 
            self.plot_policy(policy_id)  
            policy_stable = self.policy_improvement() 
            policy_id += 1 


def poisson_distribution(lamda, n):
    '''
    Parameters
    ----------
    lamda : float
        λ, the expected rate of occurrences
    n : int
        the number of occurrences

    Returns
    -------
    float
        the probability of n occurrences given λ.
    '''
    return (lamda ** n) * np.exp(-lamda) / np.math.factorial(n) if n >= 0 else 0         


if __name__ == '__main__':
    locations = [location(3, 10, lambda n : poisson_distribution(1, n), lambda n :  poisson_distribution(1, n)), location(3, 10, lambda n : poisson_distribution(1, n), lambda n :  poisson_distribution(1, n))]
    max_moving_cars_per_location = 1
    relocation_cost_per_car = [[2, 0], [0, 0]]
    gamma = 0.9
    theta = 0.001
    problem = dynamics(locations, max_moving_cars_per_location, relocation_cost_per_car, gamma, theta)   
    problem.policy_iteration()
        
    print("finished")
            

