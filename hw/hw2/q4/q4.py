import numpy as np
import matplotlib.pyplot as plt
import math
import itertools



def recommendationMatrix(R, Ob):
	O = Ob
	RO = np.dot(R, O)
	ROR = np.dot(RO, R.T)
	RORR = np.dot(ROR, R)
	res = np.dot(RORR, O)
	return res

def problem4():
	R_raw = []
	file = open("user-shows.txt")
	for raw_line in file:
		line_str = raw_line.rstrip().split(' ')
		line = map(eval, line_str)  
		R_raw.append(np.array(line))
	file.close()
	R = np.array(R_raw)
	P_diag = np.diag(np.dot(R,R.T))**-0.5
	Q_diag = np.diag(np.dot(R.T,R))**-0.5

	Q = np.diag(Q_diag)
	P = np.diag(P_diag)

	show = []
	Alex = 499
	Rec_item = recommendationMatrix(R, Q)
	Alex_item = Rec_item[Alex,:100]	
	Alex_map1 = []

	Rec_user = recommendationMatrix(R.T, P).T
	Alex_user = Rec_user[Alex,:100]
	Alex_map2 = []

	i = 0
	file = open("shows.txt")
	for raw_line in file:
		line_str = raw_line.rstrip().split('\t')
		show.append(line)
		Alex_map1.append((Alex_item[i],i,line_str))
		Alex_map2.append((Alex_user[i],i,line_str))
		i = i + 1
		if i == 100:
			break
	file.close()

	Alex_map1.sort(key=lambda l: (-l[0],l[1]))
	Alex_top5_item = Alex_map1[:5]
	for cur in Alex_top5_item:
		print str(cur[2][0])+ ": " + str(cur[0])

	Alex_map2.sort(key=lambda l: (-l[0],l[1]))
	Alex_top5_user = Alex_map2[:5]
	for cur in Alex_top5_user:
		print str(cur[2][0]) + ": " + str(cur[0])

if __name__ == '__main__':
	problem4()