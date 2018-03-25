import re
import sys
import itertools
from pyspark import SparkConf, SparkContext

def combine_pairs(word_set):
	list = []
	for cur_line in word_set:
		length = len(cur_line)
		for i in range(length):
			if cur_line[i] in L1:
				for j in range(i+1,length):
					if cur_line[j] in L1:
						if cur_line[i] < cur_line[j]:
							res = [cur_line[i],cur_line[j]]
							res.sort(key=lambda w: w)
							list.append((res[0],res[1]))
	return list

def combine_triples(word_set):
	list = []
	for cur_line in word_set:
		length = len(cur_line)
		for i in range(length):
			if cur_line[i] in L1:
				for j in range(i+1,length):
					if cur_line[j] in L1:
						for k in range(j+1,length):
							res = [cur_line[i],cur_line[j],cur_line[k]]
							res.sort(key=lambda w: w)
							list.append((res[0],res[1],res[2]))
	return list

def help_sort(line):
	sort_line = line
	sort_line.sort(key = lambda w: w)
	return sort_line

def comp_probs(l):
	(((x,y),z),p) = l 
	xy = (x,y)
	for cur in combine_set:
		if xy == cur[0]:
			dem = cur[1]
			break
	raw = (x,y,z)
	return (raw,p/float(dem))

def comp_prob_pair_x(l):
	((x,y),p) = l
	for cur in single_filtered:
		if x == cur[0]:
			dem_x = cur[1]
			break
	return ((x,y),p/float(dem_x))

def comp_prob_pair_y(l):
	((x,y),p) = l
	for cur in single_filtered:
		if y == cur[0]:
			dem_y = cur[1]
			break
	return ((y,x),p/float(dem_y))


conf = SparkConf()
sc = SparkContext(conf=conf)
data = sc.textFile('/Users/zhouzixuan/Documents/spark/hw/hw1/q2/browsing.txt')
support = 100
words = data.map(lambda line: line.rstrip().split(' '))
word_sort = words.map(lambda l: help_sort(l))
word_collect = word_sort.collect()

C1 = word_sort.flatMap(lambda l: set(l)).filter(lambda w: len(w)>0)
single = C1.map(lambda l:(l,1))
single_map = single.reduceByKey(lambda v1,v2: v1+v2)
single_filtered = single_map.filter(lambda l:l[1]>100).collect();
L1 = [n[0] for n in single_filtered]

pair_list = combine_pairs(word_collect)
pair_tmp = sc.parallelize(pair_list)
combine_pair = pair_tmp.map(lambda l:(l,1))
combine_map = combine_pair.reduceByKey(lambda v1,v2: v1+v2)
pairs_filtered = combine_map.filter(lambda l:l[1]>100)
combine_set = pairs_filtered.collect()

pairs_p_x = pairs_filtered.map(lambda l: comp_prob_pair_x(l))
pairs_p_y = pairs_filtered.map(lambda l: comp_prob_pair_y(l))
pairs_all = pairs_p_x.union(pairs_p_y).collect()
pairs_all.sort(key = lambda l: -l[1])

top5_pair = pairs_all[:5]

triple_list = combine_triples(word_collect)
triple_tmp = sc.parallelize(triple_list)
combine_triple = triple_tmp.map(lambda l:(l,1))
triple_map = combine_triple.reduceByKey(lambda v1,v2: v1+v2)
triples_filtered = triple_map.filter(lambda l:l[1]>100)

triples_x = triples_filtered.map(lambda ((a,b,c),d):(((a,b),c),d))
triple_p_x = triples_x.map(lambda l: comp_probs(l))

triples_y = triples_filtered.map(lambda ((a,b,c),d):(((a,c),b),d))
triple_p_y = triples_y.map(lambda l: comp_probs(l))

triples_z = triples_filtered.map(lambda ((a,b,c),d):(((b,c),a),d))
triple_p_z = triples_z.map(lambda l: comp_probs(l))

triple_all = triple_p_x.union(triple_p_y).union(triple_p_z).collect()
triple_all.sort(key = lambda l: (-l[1],str(l[0][0]),str(l[0][1])))
top5_triple = triple_all[:5]

f = open("out_pair_top5.txt",'w')
for i in top5_pair:
	k = str(i[0][0])
	f.write(k+"==>"+str(i[0][1])+", "+str(i[1])+"\n")
f.close()

f = open("out_triple_top5.txt",'w')
for i in top5_triple:
	k = str(i[0][0])+","+str(i[0][1])
	f.write(k+"==>"+str(i[0][2])+", "+str(i[1])+"\n")
f.close()
