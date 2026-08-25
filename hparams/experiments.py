import itertools
import json
import numpy as np

def search_lr_and_decay():
	'''
	Search the hyperparameter space of learning rate, weight decay,
	batch size, and droput rate.
	'''

	# Define search space
	combinations = None

	# Define rows
	for i,parameters in enumerate(combinations):

		sample = {}

	# Save to JSON


def benchmark_losses():
	'''
	Train best parameter choice for all loss functions across multiple seeds.
	'''
	pass


if __name__ == '__main__':
	pass