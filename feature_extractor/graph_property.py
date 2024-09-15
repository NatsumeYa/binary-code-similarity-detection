# coding=utf-8
import networkx as nx


def betweeness(g):
	"""
	计算图 g 的节点介数中心性 (betweenness centrality)
	Args:
		g (networkx.Graph): 图对象

	Returns:
		dict: 图中每个节点的介数中心性的字典 <节点，介数中心性值>
	"""
	betweenness = nx.betweenness_centrality(g)
	return betweenness


def retrieveGP(g):
	"""
	计算图 g 的节点介数中心性的平均值
	Args:
		g (networkx.Graph): 图对象

	Returns:
		float: 图 g 的节点介数中心性的平均值（保留 5 位小数）
	"""
	bf = betweeness(g)  # 计算图 g 的每个节点的介数中心性值
	x = sorted(bf.values())  # 获取所有节点的数值，并排序
	if len(x) > 0:
		value = sum(x)/len(x)  # 求平均值
	else:
		value = 0
	return round(value, 5)
