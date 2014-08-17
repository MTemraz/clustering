# Name: Mohamed Temraz
# Email: temraz11@gmail.com	
# Desription: Clustering data using kmeans

import numpy as np
import matplotlib.pyplot as plt

def preprocess(file):
    '''Format a teb-delimited dataset to a matrix'''
    data = list()
    for line in open(file):
        line = line.strip().split('\t')
        data.append(map(float,line))
    return np.mat(data)

def createCentroids(data, k):
    '''Generate random centroids'''
    rows,cols = data.shape
    centroids = np.zeros((k,cols))
    for i in range(cols):
        centroids[:,i] = np.random.uniform(data[:,i].min(),data[:,i].max(),size=k).T
    return centroids

def kMeans(data,k):
    '''Assign points to closest cluster'''
    centroids = createCentroids(data,k)
    assignments = np.zeros((data.shape[0],1))
    updated = True
    while updated:
        updated = False
        for i in range(data.shape[0]):
            current = data[i,:]
            min_dist = np.inf
            for j in range(k):
                curr_dist = euclidean(current,centroids[j,:])
                if curr_dist < min_dist:
                    min_dist = curr_dist
                    index = j
            if assignments[i,0] != index:
                assignments[i,0] = index
                updated = True
        for ind in range(k):
            pts = data[assignments==ind]
            centroids[ind,:] = np.mean(pts,axis=0)
    return assignments

def euclidean(x,y):
    '''Calculate euclidean distance between 2 vectors'''
    return np.sqrt(np.sum((a-b)**2))
