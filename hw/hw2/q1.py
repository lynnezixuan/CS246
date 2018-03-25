import scipy
from scipy import linalg
import numpy as np 

M = np.matrix([[1, 2], [2, 1], [3, 4], [4, 3]])
m = 4
n = 2
U, Sigma, V_T = scipy.linalg.svd(M, full_matrices =False)
print U
print Sigma
print V_T

if U[0][0] < 0:
	U[:,0] = np.dot(U[:,0],-1)
U_new = U[:,0]

for j in range(1,n):
	if U[0][j] < 0:
		cur = np.dot(U[:,j],-1)
	else:
		cur = U[:,j]
	U_new = np.column_stack((U_new, cur))
print U_new

Evals, Evecs = scipy.linalg.eigh(np.dot(M.T, M))

Edict = range(len(Evals))
for i in range(len(Evals)):
	Edict[i] = (Evals[i],[Evecs[i][j] for j in range(n)])

Edict_sorted =  sorted(Edict, key=lambda x: x[0], reverse=True)
print Edict_sorted

if Edict_sorted[0][1][0] < 0:
	cur = np.dot(Edict_sorted[0][1], -1)
else:
	cur = Edict_sorted[0][1]
Evecs_sorted = cur

for i in range(1,n):
	if Edict_sorted[i][1][0] < 0:
		cur = np.dot(Edict_sorted[i][1],-1)
	else:
		cur = Edict_sorted[i][1]
	Evecs_sorted = np.column_stack((Evecs_sorted, cur))
print Evecs_sorted