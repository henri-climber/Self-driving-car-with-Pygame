# Self-driving-car-with-Pygame
A car learns to drive in a small 2D environment using the NEAT-Algorithm with Pygame

# Description
A car has 5 sensors ("eyes") to detect and learn the distance to the road boundary. Using these sensors and Python's Neat library, cars are trained to steer and stay on the road. Neat is a genetic algorithm that starts with a simple neural network and increases its complexity over the next few generations. In each generation there are many cars, each car having its own neural network. Once a generation is over (all cars are "dead"), Neat constructs the neural networks of the new generation based on the neural network of the best car of the previous generation. With this technique, the algorithm learns how to drive and gets better and better over the generations.

# Modes
Run training_the_car.py to see what the training process looks like. Once training is complete, the program saves the trained version of the best car to a file.
loading_trained_car.py loads a pre-trained model and lets the car drive based on the loaded model. 

# Setup
You need to have python, neat and pygame installed in order to run the program. 
Installation of pygame: https://pypi.org/project/pygame/ 
Installation of Neat: https://pypi.org/project/neat-python/
