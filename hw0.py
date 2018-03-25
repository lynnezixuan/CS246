import re
import sys
from pyspark import SparkConf, SparkContext

conf = SparkConf()
sc = SparkContext(conf=conf)
data = sc.textFile(sys.argv[1])
words = data.flatMap(lambda line: re.split(r'\W+', line))
word_filter = words.filter(lambda word: re.match('^[A-Za-z]', word))
word_low = word_filter.map(lambda word: word.lower()).map(lambda word: word[0])
output = word_low.map(lambda c:(c,1)).reduceByKey(lambda v1, v2:v1+v2)
output.saveAsTextFile(sys.argv[2])
sc.stop()