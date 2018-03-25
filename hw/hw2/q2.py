import numpy as np
import re
import sys
import itertools
from pyspark import SparkConf, SparkContext
import matplotlib.pyplot as plt

def find_cluster(line):
	dis_list=[]
	cur_point = line
	for i in range(len(centroid)):
		cur_c = centroid[i]
		dis_e = np.linalg.norm(cur_point-cur_c)
		dis_list.append((i, cur_c, dis_e))
	sort_list = sorted(dis_list, key=lambda l: l[2])
	cost_cur = np.square(sort_list[0][2])
	res = (sort_list[0][0], 1, cur_point, cost_cur)
	return res



#initial
def k_mean(c):
	new_centroid = c
	cost_iter = [] 
	for iter in range(MAX_ITER):
		centroid = new_centroid
		cluster_map = data.map(lambda l: find_cluster(l))
		cluster_adjust = cluster_map.map(lambda (a,b,c,d):(a,(b,c,d)))
		cluster_reduce = cluster_adjust.reduceByKey(lambda v1,v2: (v1[0]+v2[0],v1[1]+v2[1],v1[2]+v2[2]))
		cluster_collect = cluster_reduce.collect()
		cost_i = np.sum([i[1][2] for i in cluster_collect])
		cost_iter.append(cost_i)
		new_centroid = [np.divide(i[1][1],i[1][0]) for i in cluster_collect]
	#[623660345.30641115, 509862908.29754496, 485480681.87200809, 463997011.68501264, 460969266.57299656, 460537847.98276806, 460313099.65354538, 460003523.88940752, 459570539.3177352, 459021103.34229112, 458490656.19198102, 457944232.58797437, 457558005.19867706, 457290136.35230219, 457050555.05956286, 456892235.61535496, 456703630.73703384, 456404203.01897502, 456177800.54199338, 455986871.0273459]
	return cost_iter

def problem2():
	conf = SparkConf()
	sc = SparkContext(conf=conf)
	raw_data = sc.textFile('/Users/zhouzixuan/Documents/spark/hw/hw2/q2/data.txt')
	#raw_data = sc.textFile('/home/zixuan95/cs246/hw/data.txt')
	raw_c1 = sc.textFile('/Users/zhouzixuan/Documents/spark/hw/hw2/q2/c1.txt')
	raw_c2 = sc.textFile('/Users/zhouzixuan/Documents/spark/hw/hw2/q2/c2.txt')
	MAX_ITER = 20
	k = 10

	data = raw_data.map(lambda l: np.array([float(i) for i in l.rstrip().split(' ')]))
	c1 = raw_c1.map(lambda l: np.array([float(i) for i in l.rstrip().split(' ')])).collect()
	c2 = raw_c2.map(lambda l: np.array([float(i) for i in l.rstrip().split(' ')])).collect()

	cost_iter_c1 = k_mean(c1)
	cost_iter_c2 = k_mean(c2)
	iteration = range(1,MAX_ITER+1)
	plt.figure(1)
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.legend("C1","C2")
    plt.plot(iteration,cost_iter_c1)
    plt.plot(iteration,cost_iter_c2)
    plt.show()


if __name__ == '__main__':
    problem2()

