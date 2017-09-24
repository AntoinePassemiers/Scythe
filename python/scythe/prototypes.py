# -*- coding: utf-8 -*-

import numpy as np
import random
from copy import deepcopy

try:
	from queue import Queue
except:
	from Queue import Queue


LEFT  = 1
RIGHT = 2
NONE  = 3

class Node:
	def __init__(self, node_id):
		self.node_id      = node_id
		self.left_child   = None
		self.right_child  = None
		self.hyperplane   = None
		self.mu           = None
		self.threshold    = None
	def whichChild(self, x):
		if self.left_child is None or self.right_child is None:
			return NONE
		elif self.hyperplane is None:
			return LEFT if (np.linalg.norm(x - self.mu) <= self.threshold) else RIGHT
		else:
			return LEFT if (np.dot(x, self.hyperplane) <= self.threshold) else RIGHT


class APDTree:
	def __init__(self, n_iterations = 4, min_n_samples = 150, n_principal_components = 5):
		self.n_features = 0
		self.densities = None
		self.n_iterations = n_iterations
		self.discretization = 100
		self.min_n_samples = min_n_samples
		self.n_principal_components = n_principal_components
		self.root = None
		self.n_nodes = 0
		self.sample_mask = None

	@staticmethod
	def computeCost(samples, decoded):
		return np.mean(np.linalg.norm(samples - decoded, axis = 1))

	def computeSphericity(self, samples):
		mu = np.mean(samples, axis = 0)
		var = np.sum(np.linalg.norm(samples - mu, axis = 1))
		return np.sqrt(2.0 * var / len(samples))

	def APDSplittingRule(self, samples):
		p = (np.random.rand(samples.shape[1]) - 0.5) * 2.0
		for i in range(self.n_iterations):
			print(X.shape, p.shape, X.shape)
			print(np.dot(X, p).shape)
			print(np.dot(np.dot(X, p).T, X).shape)
			q = np.sum(np.dot(np.dot(X, p), X))
			p = q / np.linalg.norm(q)
			print(p.shape)
		return p

	def splitNode(self, X):
		if True: # TODO : if has outliers
			mu = np.mean(X, axis = 0)
			D  = np.linalg.norm(X - mu, axis = 1)
			is_left = (D <= np.median(D))
			return is_left, mu, None, np.median(D)
		else:
			p  = self.APDSplittingRule(X)
			print(X.shape, p.shape)
			P  = np.dot(X, p)
			is_left = (P <= np.median(P))
			return is_left, None, p, np.median(P)

	def fitPCA(self, samples):
		cov = np.cov(samples.T)
		eigenvalues, eigenvectors = np.linalg.eig(cov)

		# Sort eigenvalues in decreasing order
		indices = np.argsort(eigenvalues)[::-1]
		eigenvectors = eigenvectors[:, indices]
		eigenvectors = eigenvectors[:, :self.n_principal_components]

		"""
		encoded = np.dot(samples, eigenvectors)
		decoded = np.dot(encoded, eigenvectors.T)
		"""
		
		cost = self.computeSphericity(samples)
		return cost, eigenvectors

	def preprocessDensities(self, X):
		self.n_features = X.shape[1]
		self.densities = list()
		for f in range(self.n_features):
			bins = np.arange(self.discretization)
			self.densities.append(np.digitize(X[:, f], bins))
		boundaries = [[0, len(self.densities[f])] for f in range(self.n_features)]
		return boundaries

	def evaluateFeature(self, feature_id, X, bounds):
		if bounds[0] == bounds[1]:
			return np.inf, 0, None
		split = self.densities[feature_id][random.choice(np.arange(bounds[0], bounds[1]))]
		left_samples  = X[X[:, feature_id] <= split]
		right_samples = X[X[:, feature_id] >  split]
		n_left, n_right = len(left_samples), len(right_samples)
		if n_left <= self.min_n_samples or n_right <= self.min_n_samples:
			return np.inf, 0, None
		left_cost, left_pca = self.fitPCA(left_samples)
		right_cost, right_pca = self.fitPCA(right_samples)

		cost = n_left * left_cost ** 2 + n_right * right_cost ** 2
		return cost, split, (left_pca, right_pca)

	def fit(self, X):
		boundaries = self.preprocessDensities(X) #TODO
		n_samples = X.shape[0]
		self.sample_mask = np.zeros(n_samples, dtype = np.int)
		self.root = Node(0)
		self.n_nodes = 1
		queue = Queue()
		queue.put(self.root)

		while not queue.empty():
			current_node = queue.get()
			samples = X[self.sample_mask == current_node.node_id]

			is_left, mu, hyperplane, threshold = self.splitNode(samples)
			current_node.hyperplane = hyperplane
			current_node.mu = mu
			current_node.threshold = threshold

			cost, eigenvectors = self.fitPCA(samples)
			current_node.eigenvectors = eigenvectors

			if len(samples) >= self.min_n_samples:
				left_mask = np.zeros(len(X), dtype = np.bool)
				left_mask[self.sample_mask == current_node.node_id] = is_left
				right_mask = np.zeros(len(X), dtype = np.bool)
				right_mask[self.sample_mask == current_node.node_id] = ~is_left
				self.sample_mask[left_mask]  = self.n_nodes
				self.sample_mask[right_mask] = self.n_nodes + 1

				current_node.left_child  = Node(self.n_nodes)
				current_node.right_child = Node(self.n_nodes + 1)
				queue.put(current_node.left_child)
				queue.put(current_node.right_child)
				self.n_nodes += 2

	def encode_with_root(self, X):
		return np.dot(X, self.root.eigenvectors)

	def decode_with_root(self, X):
		return np.dot(X, self.root.eigenvectors.T)

	def encode(self, X):
		X = np.asarray(X)
		n_samples = X.shape[0]
		result = np.empty((n_samples, self.n_principal_components), dtype = X.dtype)
		for i in range(n_samples):
			current_node = self.root
			while current_node is not None:
				child = current_node.whichChild(X[i, :])
				if child == NONE:
					break
				elif child == LEFT:
					current_node = current_node.left_child
				else:
					current_node = current_node.right_child
			eigvecs = current_node.eigenvectors
			# eigvecs = self.root.eigenvectors
			result[i, :] = np.dot(X[i, :], eigvecs)
		return result

	def decode(self, X):
		X = np.asarray(X)
		n_samples = X.shape[0]
		result = np.empty((n_samples, self.n_features), dtype = X.dtype)
		for i in range(n_samples):
			current_node = self.root
			while True:
				d = np.dot(X[i, :], current_node.eigenvectors.T)
				child = current_node.whichChild(d)
				if child == NONE:
					break
				elif child == LEFT:
					current_node = current_node.left_child
				else:
					current_node = current_node.right_child
			# eigvecs = self.root.eigenvectors
			eigvecs = current_node.eigenvectors
			result[i, :] = np.dot(X[i, :], eigvecs.T)
		return result


if __name__ == "__main__":
	from sklearn.decomposition import PCA
	from sklearn.datasets import make_classification
	import matplotlib.pyplot as plt

	from scythe.MNIST import *

	n_samples  = 5000
	n_features = 20
	n_classes  = 10
	X, y = make_classification(
		n_samples     = 2 * n_samples, 
		n_informative = 7,
		n_features    = 20,
		n_classes     = 10)

	tree = APDTree(n_principal_components = 2, min_n_samples = 500, n_iterations = 8)
	tree.fit(X)
	print(tree.n_nodes)
	sc_encoded = tree.encode_with_root(X)
	sc_decoded = tree.decode_with_root(sc_encoded)
	print("Scythe encoding error : %f" % APDTree.computeCost(X, sc_decoded))
	
	pca = PCA(n_components = 2)
	pca.fit(X)
	sk_encoded = pca.transform(X)
	sk_decoded = pca.inverse_transform(sk_encoded)
	print("Sklearn encoding error: %f" % APDTree.computeCost(X, sk_decoded))

	def scatterPlot(encoded, title = ""):
		colors = ["red", "yellow", "cyan", "black", "brown", "blue", "green", "purple", "orange", "orangered"]
		for c in range(n_classes):
			samples = encoded[y == c, :]
			plt.scatter(samples[:, 0], samples[:, 1], c = colors[c])
			plt.xlabel("Principal component 1")
			plt.ylabel("Principal component 2")
			plt.title(title)

	plt.figure(1)
	plt.subplot(121)
	scatterPlot(sc_encoded, "Scythe APD tree")
	plt.subplot(122)
	scatterPlot(sk_encoded, "Scikit-learn PCA")
	plt.show()

	print("Finished")