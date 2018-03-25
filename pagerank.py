import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
data = sc.textFile('/Users/zhouzixuan/Documents/spark/graph-full.txt')
raw_line = data.map(lambda line: line.rstrip().split('\t'))
line = raw_line.map(lambda line: (line[0],line[1])).distinct()
relation = line.groupByKey()
n = relation.count()
r = relation.map(lambda (a,b):(a, 1./n))

def computeMr(neighbor, rank):
	res = []
	deg = len(neighbor)
	for cur in neighbor:
		res.append((cur, rank/deg))
	return res

for iter in range(40):
	Mr_temp = relation.join(r)
	Mr = Mr_temp.flatMap(lambda (a,(b,c)): computeMr(b,c))
	r_temp = Mr.reduceByKey(lambda v1, v2: v1 + v2)
	r = r_temp.map(lambda (a,b): (a, 0.2/n + 0.8 * b))

bottom = r.sortBy(lambda pair: pair[1]).take(5)
top = r.sortBy(lambda pair: -pair[1]).take(5)
print "bottom"
print bottom
print "top"
print top