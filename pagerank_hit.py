import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
data = sc.textFile('/Users/zhouzixuan/Documents/spark/graph-full.txt')
raw_line = data.map(lambda line: line.rstrip().split('\t'))
L = raw_line.map(lambda line: (line[0],line[1])).distinct().groupByKey()
L_transpose = raw_line.map(lambda line: (line[1],line[0])).distinct().groupByKey()
h = L.map(lambda (a,b):(a, 1))
# 2(a)
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

# 2(b)
def computeLh_La(Ls, h):
	res = []
	for l in Ls:
		res.append((l, h))
	return res

for iter in range(40):
	Lh_temp = L.join(h)
	Lh = Lh_temp.flatMap(lambda (a,(b,c)): computeLh_La(b,c))
	a_raw = Lh.reduceByKey(lambda v1, v2: v1 + v2)
	a_max = max([e[1] for e in a_raw.collect()])
	a = a_raw.map(lambda (a,b):(a, b/float(a_max)))
	La_temp = L_transpose.join(a)
	La = La_temp.flatMap(lambda (a,(b,c)): computeLh_La(b,c))
	h_raw = La.reduceByKey(lambda v1, v2: v1 + v2)
	h_max = max([e[1] for e in h_raw.collect()])
	h = h_raw.map(lambda (a,b):(a, b/float(h_max)))

bottom_a = a.sortBy(lambda pair: pair[1]).take(5)
top_a = a.sortBy(lambda pair: -pair[1]).take(5)
print "bottom_authority"
print bottom_a
print "top_authority"
print top_a

bottom_h = h.sortBy(lambda pair: pair[1]).take(5)
top_h = h.sortBy(lambda pair: -pair[1]).take(5)
print "bottom_hubbiness"
print bottom_h
print "top_hubbiness"
print top_h