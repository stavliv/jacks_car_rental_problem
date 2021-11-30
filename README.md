# jacks_car_rental_problmem

This is a modified version of the Jack's car rental problem as stated in Reinforcement Learning An Introduction by Richard S. Sutton and Andrew G. Barto, so that it can work with any number of locations, with costumized problem settings (rewards, costs, maximum number of cars, maximum number of cars moved, gamma and car requests and returns distributions), solved through policy iteration.

## problem statement in Reinforcement Learning An Introduction by Richard S. Sutton and Andrew G. Barto
Jack manages two locations for a nationwide car rental company. Each day, some number of customers arrive at each location to rent cars.
If Jack has a car available, he rents it out and is credited $10 by the national company.
If he is out of cars at that location, then the business is lost. 
Cars become available for renting the day after they are returned. 
To help ensure that cars are available where they are needed, Jack can move them between the two locations overnight, at a cost of $2 per car moved. 
We assume that the number of cars requested and returned at each location are Poisson random variables,
To simplify the problem slightly, we assume that there can be no more than 20 cars at each location (any additional cars
are returned to the nationwide company, and thus disappear from the problem) and a maximum of five cars can be moved from one location to the other in one night.
We take the discount rate to be γ = 0.9 and formulate this as a continuing finite MDP, where the time steps are days, 
the state is the number of cars at each location at the end of the day, and the actions are the net numbers of cars moved between the two locations overnight.
