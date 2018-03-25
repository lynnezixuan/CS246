import numpy as np
import math
import time
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

file_y = open("target.txt")
raw_Y = []
for line in file_y:
    raw_Y.append(int(line.strip()))
m = len(raw_Y)
Y = np.array(raw_Y).reshape((m, 1))

file_x = open("features.txt")
raw_X = []
for line in file_x:
    tmp = line.strip().split(",")
    tmp = map(int, tmp)
    raw_X.append(tmp)
X = np.array(raw_X)
n = X.shape[1]
X, Y = shuffle(X, Y, random_state=0)

#batch 
k = 0
eta = 0.0000003
e = 0.25
w = np.zeros((n,1))
b = 0
C = 100
cost_new = float("Inf")
cost_old = float("Inf")
cost_list_batch = []
start = time.time()
while True:
    cost_part1 = 1./2 * np.sum(w*w)
    r = Y * (np.dot(X, w) + b)
    mask_cost = (1-r) > 0
    cost_part2 = np.sum((1 - r) * mask_cost,axis = 0, keepdims = True)
    cost_new = cost_part1 + C * cost_part2

    mask = r < 1 # 6000 * 1
    grad_raw = -Y * X # 6000 * 122
    grad_part2 = np.sum(grad_raw * mask, axis = 0, keepdims = True).T
    grad = w + C * grad_part2
    w = w - eta * grad
    grad_raw_b = -Y
    grad_part2_b = np.sum(grad_raw_b * mask, axis = 0, keepdims = True).T
    grad_b = C * grad_part2_b
    b = b - eta * grad_b
    k = k + 1

    conv_cost = np.abs(cost_old - cost_new) * 100 / cost_old
    if conv_cost < e: break     
    cost_old = cost_new
    cost_list_batch.append(cost_old[0][0])
print k
print time.time() - start
t1 = k

#SGD
k = 0
eta = 0.0001
e = 0.001
w = np.zeros((n,1))
b = 0
C = 100
cost_new = 0
cost_old = 0
conv_cost = 0
i = 1
# np.random.shuffle(X)
cost_list_sgd = []
start = time.time()
while True:
    cost_part1 = 1./2 * np.sum(w*w)
    r = Y[i-1] * (np.dot(X[i-1].reshape(1, 122), w) + b)  
    r_cost = Y * (np.dot(X, w) + b)
    mask_cost = (1-r_cost) > 0
    cost_part2 = np.sum((1 - r_cost) * mask_cost,axis = 0, keepdims = True)
    cost_new = cost_part1 + C * cost_part2
    
    mask = r < 1 # 6000 * 1
    grad_raw = -Y[i-1] * (X[i-1].reshape(1, 122))# 6000 * 122
    grad_part2 = (grad_raw * mask).T
    grad = w + C * grad_part2
    w = w - eta * grad
    grad_raw_b = -Y[i-1]
    grad_part2_b = grad_raw_b * mask
    grad_b = C * grad_part2_b
    b = b - eta * grad_b
    
    if k > 0:
        conv_cost_new = np.abs(cost_old - cost_new) * 100 / cost_old
        conv_cost = 0.5 * conv_cost + 0.5 * conv_cost_new       
        if conv_cost < e: break 
    k = k + 1
    i = i % m + 1
    cost_old = cost_new
    cost_list_sgd.append(cost_old[0][0])
print k
print cost_new
print time.time() - start
t2 = k

#mini batch
k = 0
l = 0
eta = 0.00001
e = 0.01
batch_size = 20
w = np.zeros((n,1))
b = 0
C = 100
cost_new = 0
cost_old = 0
conv_cost = 0
cost_list_mini = []
start = time.time()
while True:
    cost_part1 = 1./2 * np.sum(w*w)
    r = Y[l * batch_size : (l+1) * batch_size] * (np.dot(X[l * batch_size : (l+1) * batch_size], w) + b)  
    r_cost = Y * (np.dot(X, w) + b)
    mask_cost = (1-r_cost) > 0
    cost_part2 = np.sum((1 - r_cost) * mask_cost,axis = 0, keepdims = True)
    cost_new = cost_part1 + C * cost_part2
    
    mask = r < 1 # 6000 * 1
    grad_raw = -Y[l * batch_size : (l+1) * batch_size] * (X[l * batch_size : (l+1) * batch_size])# 6000 * 122
    grad_part2 = np.sum(grad_raw * mask, axis = 0, keepdims = True).T
    grad = w + C * grad_part2
    w = w - eta * grad
    grad_raw_b = -Y[l * batch_size : (l+1) * batch_size]
    grad_part2_b = np.sum(grad_raw_b * mask, axis = 0, keepdims = True).T
    grad_b = C * grad_part2_b
    b = b - eta * grad_b
    if k > 0:
        conv_cost_new = np.abs(cost_old - cost_new) * 100 / cost_old
        conv_cost = 0.5 * conv_cost + 0.5 * conv_cost_new       
        if conv_cost < e: break 
    k = k + 1
    l = (l + 1) % ((m + batch_size - 1)/batch_size)
    cost_old = cost_new
    cost_list_mini.append(cost_old[0][0])
print k
print cost_new
print time.time() - start
t3 = k

plt.figure()
plt.hold(True)
iter1 = range(t1 - 1)
iter2 = range(t2)
iter3 = range(t3)
plt.xlabel("Iteration")
plt.ylabel("Cost")
label = ["Batch GD","SGD","mini batch GD"]
plt.title("Cost vs Iteration")
a1 = plt.plot(iter1, cost_list_batch)
a2 = plt.plot(iter2, cost_list_sgd)
a3 = plt.plot(iter3, cost_list_mini)
plt.legend(label,loc = 0, ncol = 2)
plt.show()
