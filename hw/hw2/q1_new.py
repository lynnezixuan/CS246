import scipy
from scipy import linalgs
import numpy as np 

M = np.matrix([[1, 2], [2, 1], [3, 4], [4, 3]])
m = 4
n = 2
U, Sigma, V_T = scipy.linalg.svd(M, full_matrices =False)
print U
print Sigma
print V_T

Evals, Evecs = scipy.linalg.eigh(np.dot(M.T, M))

Edict = range(len(Evals))
for i in range(len(Evals)):
	Edict[i] = (Evals[i],[Evecs[i][j] for j in range(n)])

Edict_sorted =  sorted(Edict, key=lambda x: x[0], reverse=True)
print Edict_sorted

cur = Edict_sorted[0][1]
Evecs_sorted = cur

for i in range(1,n):
	cur = Edict_sorted[i][1]
	Evecs_sorted = np.column_stack((Evecs_sorted, cur))
print Evecs_sorted