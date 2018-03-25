import math
import matplotlib.pyplot as plt

def hash_fun(a, b, p, n_buckets, x):
    y = x % p
    hash_val = (a * y + b) % p
    return hash_val % n_buckets

hash_file = "/Users/zhouzixuan/Documents/spark/hw/hw4/Hw4-q4/hash_params.txt"
hash_raw = open(hash_file)
a = []
b = []
for line in hash_raw:
    par = line.strip().split("\t")
    a.append(int(par[0]))
    b.append(int(par[1]))

count_file = "/Users/zhouzixuan/Documents/spark/hw/hw4/Hw4-q4/counts.txt"
counts_raw = open(count_file)
counts = []
for line in counts_raw:
    par = line.strip().split("\t")
    counts.append(int(par[1]))

word_file = "/Users/zhouzixuan/Documents/spark/hw/hw4/Hw4-q4/words_stream.txt"
p = 123457
e = math.exp(1) * 0.0001
delta = math.exp(-5)
hash_num = 5
hash_fun_list = []
for i in range(hash_num):
    hash_fun_list.append({})

len_stream = 0
n_buckets = int(math.ceil(math.exp(1)/e))
word_stream = open(word_file)
for line in word_stream:
    len_stream = len_stream + 1
    raw_num = line.strip().split("\t")
    x = int(raw_num[0])
    for i in range(hash_num):
        hash_val = hash_fun(a[i], b[i], p, n_buckets, x)
        if hash_val not in hash_fun_list[i]:
            hash_fun_list[i][hash_val] = 1
        else:
            hash_fun_list[i][hash_val] = 1 + hash_fun_list[i][hash_val]

error_list = []
x_list = []
for i in range(len(counts)):
    F_i = counts[i]
    x = i + 1
    c_list = []
    for j in range(hash_num):
        hash_val = hash_fun(a[j], b[j], p, n_buckets, x)
        c_tmp = hash_fun_list[j][hash_val]
        c_list.append(c_tmp)
    c_min = min(c_list)
    error_tmp = abs(c_min - F_i)/float(F_i)
    x_list.append(F_i/float(len_stream))
    error_list.append(error_tmp)

ax = plt.gca()
ax.plot(x_list,error_list,'b.') 
ax.set_yscale('log')
ax.set_xscale('log')
plt.title('The result of algorithm: frequency vs relative error') 
plt.xlabel("Word frequency")
plt.ylabel("Relative error")
plt.show()







