import re
import sys
import itertools
from pyspark import SparkConf, SparkContext

def transform(line):
	list = []
	for i in line[1]:
		list.append((line[0],i))
	return list

def combine_pair(line):
	list = []
	for i in line[1]:
		for j in line[1]:
			if i == j: continue
			list.append((i,j))
	return list

def sorted_friend(line):
	if len(line) == 2:
		sl = line[1]
		sl.sort(key=lambda l: (-l[1], int(l[0])))
		if len(sl) > 10:
			sl = sl[:10]
		recommend_name = [int(i[0]) for i in sl] 
	return (line[0], recommend_name)

def testValue(line):
	if line[0] in [924, 8941, 8942, 9019, 9020, 9021, 9022, 9990, 9992, 9993]:
		return True
	else:
		return False

conf = SparkConf()
sc = SparkContext(conf=conf)
data = sc.textFile('/Users/zhouzixuan/Documents/spark/hw/hw1/q1/soc-LiveJournal1Adj.txt')
words = data.flatMap(lambda line: line.split(' '))
fw = words.map(lambda l: l.split('\t'))
f_word = fw.mapValues(lambda w: re.split(r'\W+', w))
original_map = f_word.map(lambda w: (w[0],[]))
f_word_filtered= f_word.filter(lambda w: len(w)==2)
direct_friend = f_word_filtered.flatMap(lambda l:transform(l))
dfs = direct_friend.map(lambda w: (w,0))
common_pair = f_word.flatMap(lambda l:combine_pair(l))
cps = common_pair.map(lambda w: (w,1))
tp = cps.reduceByKey(lambda v1,v2: v1+v2)
tpp = tp.subtractByKey(dfs)
cp = tpp.map(lambda ((a,b),c):(a,(b,c)))
grouped_map = cp.groupByKey().mapValues(lambda l: list(l))
sorted_map = grouped_map.map(lambda l: sorted_friend(l)) 
empty = original_map.subtractByKey(sorted_map)
complete_map = sorted_map.union(empty)
out_map = complete_map.map(lambda l:(int(l[0]),l[1])).sortByKey()
op = out_map.collect()

f = open("out_complete.txt",'w')
for i in op:
	k = ",".join([str(n) for n in i[1]])
	f.write(str(i[0])+" "+k+"\n")
f.close()

opp = out_map.filter(lambda l:testValue(l)).collect()
ff = open("out_specific.txt",'w')
for i in opp:
	kk = ",".join([str(n) for n in i[1]])
	ff.write(str(i[0])+" "+kk+"\n")
ff.close()

out_map.saveAsTextFile('/Users/zhouzixuan/Documents/spark/hw/hw1/out.txt')
sc.stop()








