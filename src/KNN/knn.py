# -*- coding:utf-8 -*-  
# 电影名称	打斗次数	接吻次数	电影类型
# California Man 3	104	Romance
# He’s Not Really into Dudes 2	100	Romance
# Beautiful Woman 1	81	Romance
# Kevin Longblade 101	10	Action
# Robo Slayer 3000 99	5	Action
# Amped II 98	2	Action
# 未知	18	90	Unknown
def KNN(data,test_data,K=3):
	import pdb
	labels=data[:,-1]
	#
	data=data[:,:-1]
	dis=[]
	for each_data in data:
		distance=(test_data-each_data)**2
		distance=distance.sum()*0.5
		dis.append(distance)
	#pdb.set_trace()
	dis_sorted=sorted(dis)
	# 找到前K个
	K_dis=dis_sorted[0:K]
	# 在dis中找到对应的前K个元素的下标
	K_index=[]
	for each_dis in K_dis:
		K_index.append(dis.index(each_dis))
	# 计算labels中统计次数
	K_labels=labels[K_index]
	from collections import Counter
	c=Counter(K_labels).most_common()[:1]
	print "test_data should to be judge to be :",c[0][0]


if __name__ == '__main__':
    import numpy as np
    data=np.array([[3,104,0],[2,100,0],[1,81,0],[101,10,1],[99,5,1],[98,2,1]])
    test_data=np.array([90,3])
    KNN(data,test_data,3)