#-*-coding:utf-8-*-

# 这是决策树ID3的代码实现
import numpy as np
import math
def createDataSet():
	#dataset=np.array([[1,1,1],[1,1,1],[1,0,0],[0,1,0],[0,1,0]])
	#dataset=[[1,1,1],[1,1,1],[1,0,0],[0,1,0],[0,1,0]]
	dataset=[[1,1,'Y'],[1,1,'Y'],[1,0,'N'],[0,1,'N'],[0,1,'N']]
	features=['No surfacing','flippers']
	return dataset,features


def calEntropy(dataset):
	data_length=len(dataset)
	labelCounts={}
	for data in dataset:
		currentLabel=data[-1]
		# 当currentLabel在labelCounts中不存在时，初始化0
		if currentLabel not in labelCounts.keys():
			labelCounts[currentLabel]=0
		labelCounts[currentLabel]+=1
	Entropy=0.0
	for key in labelCounts:
		p=float(labelCounts[key])/data_length
		if p!=0:
			Entropy-=p*math.log(p,2)
	return Entropy

def chooseBestFeature(dataset):
	numFeatures=len(dataset[0])-1
	baseEntropy=calEntropy(dataset)
	bestInfoAdd=0.0
	BestFeature=-1
	for i in range(numFeatures):
		featValues=[example[i] for example in dataset]
		# 去重，每种只留一个值
		uniqueFeatValues=set(featValues)
		newEntropy=0.0
		splitInfor=0.0
		for val in uniqueFeatValues:
			subDataSet=splitDataSet(dataset,i,val)
			p=len(subDataSet)/float(len(dataset))
			newEntropy+=p*calEntropy(subDataSet)
			splitInfor+=abs(p*math.log(p,2))
		#pdb.set_trace()
		if(baseEntropy-newEntropy)/splitInfor >bestInfoAdd:
			bestInfoAdd=(baseEntropy-newEntropy)/splitInfor 
			BestFeature=i
		# if(baseEntropy-newEntropy)>bestInfoGain:
		# 	bestInfoGain=baseEntropy-newEntropy
		# 	BestFeature=i
	return BestFeature

# def splitDataSet(dataset,BestFeature,values):
# 	retDataset=np.array([])
# 	index=-1
# 	index_list=[]
# 	for featvec in dataset:
# 		index+=1
# 		if featvec[BestFeature]==values:
# 			index_list.append(index)
# 	retDataset=dataset[index_list]
# 	return retDataset
def splitDataSet(dataset,feat,values):
	retDataSet = []
	# 以下
	for featVec in dataset:
		if featVec[feat] == values:
			reducedFeatVec = featVec[:feat]
			reducedFeatVec.extend(featVec[feat+1:])
			retDataSet.append(reducedFeatVec)
	return retDataSet

def treeGrowth(dataSet,features):
	# 获取类别信息
    classList = [example[-1] for example in dataSet]  
    if classList.count(classList[0])==len(classList):
    	# print '怎么只有一个类别信息？？？数据一定出错了，亲！！！'
        return classList[0] 
    # if len(dataSet[0])==1:
    # 	return classify(classList)

    # 找到信息增益最大的那个特征（区分能力最强），返回的是features的下标
    bestFeat = chooseBestFeature(dataSet)
    #pdb.set_trace()
    bestFeatLabel = features[bestFeat]
    myTree = {bestFeatLabel:{}}  
    featValues = [example[bestFeat] for example in dataSet]  
    uniqueFeatValues = set(featValues)  
    del (features[bestFeat])  
    for values in uniqueFeatValues:
    	#pdb.set_trace()
        subDataSet = splitDataSet(dataSet,bestFeat,values)
        myTree[bestFeatLabel][values] = treeGrowth(subDataSet,features)  
    features.insert(bestFeat, bestFeatLabel)  
    return myTree

def predict(tree,newObject):
	while isinstance(tree,dict):
		key=tree.keys()[0]
		#pdb.set_trace()
		tree=tree[key][newObject[key]]
	return tree

if __name__=='__main__':
	import pdb
	dataset,features=createDataSet()
	tree=treeGrowth(dataset,features)
	print tree
	test={'No surfacing':1,'flippers':1}
	print predict(tree,test)



