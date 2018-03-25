import numpy as np
import matplotlib.pyplot as plt
import math

def Initialization_parameter(file_name):
	user_max = 0
	movie_max = 0
	file = open(file_name)
	for raw_line in file:
		line = raw_line.rstrip().split('\t')
		user, movie, rating = (int(line[0]), int(line[1]), int(line[2]))
		user_max = max(user, user_max)
		movie_max = max(movie, movie_max)
	file.close()
	return (user_max, movie_max)

def draw(MAX_ITER, error_list, num):
	plt.figure(num)
	plt.xlabel("Iteration")
	plt.ylabel("Error")
	iteration = range(MAX_ITER)
	plt.plot(iteration, error_list)
	plt.show()

def problem3():

	MAX_ITER = 40
	k = 20
	learning_rate = 0.02# 62231.9076897
	# 0.01 51062.9932995
	# 0.02 54411.4750902
	lambdas = 0.1
	n, m = Initialization_parameter("ratings.train.txt")
	Q = np.random.rand(m, k) * math.sqrt(5/float(k))
	P = np.random.rand(n, k) * math.sqrt(5/float(k))
	error_list = []
	for iter in range(MAX_ITER):
		error_1 = 0
		set_n = set()
		set_m = set()
		file = open("ratings.train.txt")
		for raw_line in file:
			line = raw_line.rstrip().split('\t')
			usr, mov, rating = (int(line[0]), int(line[1]), int(line[2]))
			user = usr - 1
			movie = mov - 1
			set_n.add(user)
			set_m.add(movie)
			epsilon = 2 * (rating - np.dot(Q[movie,:], P[user,:].T))
			q_old = Q[movie,:] + learning_rate * (epsilon * P[user, :] - 2 * lambdas * Q[movie,:])
			p_old = P[user, :] + learning_rate * (epsilon * Q[movie,:] - 2 * lambdas * P[user, :])
			Q[movie,:] = q_old
			P[user, :] = p_old
		file.close()

		file = open("ratings.train.txt")
		for raw_line in file:
			line = raw_line.rstrip().split('\t')
			usr, mov, rating = (int(line[0]), int(line[1]), int(line[2]))
			user = usr - 1
			movie = mov - 1
			tmp = rating - np.dot(Q[movie,:], P[user,:].T)	
			error_1 = error_1 + tmp**2
		file.close()

		user_sum = 0
		movie_sum = 0
		rate_n = list(set_n)
		rate_m = list(set_m)
		for i in rate_n:
			user_sum = user_sum + np.linalg.norm(P[i, :])**2

		for j in rate_m:
			movie_sum = movie_sum + np.linalg.norm(Q[j, :])**2

		error = error_1 + lambdas * (user_sum + movie_sum)
		error_list.append(error)

	print(error_list[39])
	draw(MAX_ITER, error_list, 1)

if __name__ == '__main__':
	problem3()
