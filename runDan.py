#%matplotlib inline
import os
import random
import sys
import time
import pickle
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pyximport
pyximport.install(reload_support=True)
import skm
print("Hello, world, I'm starting")
M, N = 10000, 1000

try:
  with open("data_train.pickle", "rb") as filor:
    data, test = pickle.load(filor)

except FileNotFoundError:
  def parse_line(line):
    coord, rating = line.split(',')
    user, film = coord.split('_')
    return (int(user[1:]), int(film[1:]), int(rating))
    
  data = sp.lil_matrix((M, N))
  test = sp.lil_matrix((M, N))
  with open("data_train.csv") as filou:
    filou.__next__() # skip header
    for line in filou:
        user, film, rating = parse_line(line)
        if random.randrange(10) == 0:
          test[user-1, film-1] = rating
        else:
          data[user-1, film-1] = rating
        
  data = data.tocsr()
  test = test.tocsr()
  with open("data_train.pickle", mode="bw") as filor:
    pickle.dump((data, test), filor)

def rmse(test, guess):
  diffs = test - test.sign().multiply(guess)
  return np.sqrt(np.sum(np.square(diffs.data)) / diffs.nnz)

#The slow version
def k_means(data, test, K=10, tol=0.01, centroids=None):
  M, N = data.shape
  def get_assignments(centroids):
    assignments = np.zeros(M, dtype=np.int32)
    for user in range(M):
      dists = np.linalg.norm(data[user] - data[user].sign().multiply(centroids), axis=1)
      assignments[user] = np.argmin(dists)
    return assignments
    
  def get_centroids(assignments):
    centroids = np.zeros((K, N))
    counts = np.zeros((K, N))
    for k in range(K):
      centroids[k] = data[assignments == k].sum(0)
      counts[k] = data[assignments == k].sign().sum(0)
    counts[counts == 0] = 1
    return centroids / counts
    
  err = 999
    
  if centroids is None:
    centroids = np.random.rand(K, N) * 6 + 0.5
  while True:
    assignments = get_assignments(centroids)
    centroids = get_centroids(assignments)
    newerr = rmse(test, centroids[assignments])
    if (err - newerr) / err < tol:
      break
    err = newerr
  return err, centroids, assignments

#Regular k-means
def dense_k_means(sparse_data, test, K=10, tol=0.01, centroids=None):
  M, N = sparse_data.shape
  data = sparse_data.todense().A
  data[data == 0] = sparse_data.sum() / sparse_data.nnz
    
  def get_assignments(centroids):
    assignments = np.zeros(M, dtype=np.int32)
    for user in range(M):
      dists = np.linalg.norm((data[user] - centroids), axis=1)
      assignments[user] = np.argmin(dists)
    return assignments

  def get_centroids(assignments):
    centroids = np.zeros((K, N))
    for k in range(K):
      centroids[k] = np.sum(data[assignments == k], axis=0)
      counts = np.bincount(assignments, minlength=K)
      counts[counts == 0] = 1
      return centroids / counts[:,np.newaxis]
    #def rmse(guess):
    #    return np.sqrt(np.sum(np.square(guess - test)*np.sign(test)) / np.sum(np.sign(test)))
    
  err = 999
    
  if centroids is None:
    centroids = np.random.rand(K, N) * 6 + 0.5
    
    
  while True:
    assignments = get_assignments(centroids)
    centroids = get_centroids(assignments)
    newerr = rmse(test, centroids[assignments])
    if (err - newerr) / err < tol:
      break
    err = newerr
  return err, centroids, assignments

errs = []
for k in range(2, 10):
  err = min(skm.k_means(data, test, K=k, tol=0.001)[0] for run in range(6))
  errs.append(err)
print(min(errs))
plt.plot(range(2, 10), errs)
plt.show()

errs = []
for k in range(2, 20):
  err = min(dense_k_means(data, test, K=k, tol=0.001)[0] for run in range(4))
  errs.append(err)
print(min(errs))
plt.plot(range(2, 20), errs)
plt.show()

err, centroids, assignments = skm.k_means(data, test, K=4, tol=0.0001)
data1 = data.todense()
data1[data1 == 0] = 3.66
errs = []
U, s, V = np.linalg.svd(np.matrix(data1))
for k in range(2, 20):
  guess = np.array(U[:,0:k] * np.diag(s[0:k]) * V[0:k,:])
  errs.append(rmse(test, guess))
print(min(errs))
plt.plot(range(2, 20), errs)
plt.show()

data1, U, s, V, guess = None, None, None, None, None


data2 = data.todense()
data2 = np.where(data2 != 0, data2, centroids[assignments])
errs = []
U, s, V = np.linalg.svd(np.matrix(data2))
for k in range(2, 30):
  guess = np.array(U[:,0:k] * np.diag(s[0:k]) * V[0:k,:])
  errs.append(rmse(test, guess))
print(min(errs))
plt.plot(range(2, 30), errs)
plt.show()
