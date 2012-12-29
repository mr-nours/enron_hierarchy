from tulip import *
import math

def main(graph): 
	nodeDegree =  graph.getDoubleProperty("nodeDegree")
	nodeCentrality =  graph.getDoubleProperty("nodeCentrality")
	cliqueNumber =  graph.getIntegerProperty("cliqueNumber")	
	rawUserCliqueScore = graph.getDoubleProperty("rawUserCliqueScore")
	weightedUserCliqueScore = graph.getDoubleProperty("weightedUserCliqueScore")
	clusteringCoefficient = graph.getDoubleProperty("clusteringCoefficient")
	shortestPath = graph.getDoubleProperty("shortestPath")
	timeScore = graph.getDoubleProperty("timeScore")
	liste = list(find_cliques(graph))

	# Structural indicators
	for n in graph.getNodes():	
		nbOccurence = compute_cliqueNumber(liste,n)
		cliqueNumber.setNodeValue(n,nbOccurence)	
		CliqueScore = compute_RawCliqueScore(liste,n)
		rawUserCliqueScore.setNodeValue(n, CliqueScore)
		weightedScore = compute_weightedCliqueScore(liste,n,timeScore.getNodeValue(n))
		
	#Assigns to each node its local clustering coefficient	
	tlp.clusteringCoefficient(graph,clusteringCoefficient,graph.numberOfNodes())

	shortestPathLength = tlp.averagePathLength(graph)
	graph.computeDoubleProperty("Degree", nodeDegree)
	graph.computeDoubleProperty("Betweenness Centrality", nodeCentrality)

	dataSet = tlp.DataSet()
	dataSet["closeness centrality"] = True
	dataSet["norm"] = False
	graph.computeDoubleProperty("Eccentricity", shortestPath, dataSet)

	# Normalisation des indicateurs sur une echelle [0-100]
	NormalizeMetricValue(graph,cliqueNumber)
	NormalizeMetricValue(graph,rawUserCliqueScore)
	#NormalizeMetricValue(graph,weightedUserCliqueScore)
	NormalizeMetricValue(graph,nodeDegree)
	NormalizeMetricValue(graph,clusteringCoefficient)
	NormalizeMetricValue(graph,nodeCentrality)

def find_cliques(graph):
	#Cache nbrs and find first pivot (highest degree)
	maxconn=-1
	nnbrs={}
	pivotnbrs=set() # handle empty graph
	for n in graph.getNodes():
		nbrs=set(graph.getInOutNodes(n))
		nbrs.discard(n)
		conn = len(nbrs)
		if conn > maxconn:
			nnbrs[n] = pivotnbrs = nbrs		
			maxconn = conn	
		else :
			nnbrs[n] = nbrs 

	# Initial setup
	cand=set(nnbrs)
	smallcand = set(cand - pivotnbrs) 
	done=set()
	stack=[]
	clique_so_far=[]
	# Start main loop
	while smallcand or stack:
		try:
			# Any nodes left to check?
			n=smallcand.pop()
		except KeyError:
			# back out clique_so_far
			cand,done,smallcand = stack.pop()
			clique_so_far.pop()
			continue
          
		# Add next node to clique
		clique_so_far.append(n)
		cand.remove(n) #on supprimer le noeud pivot de la liste
		done.add(n)
		nn=nnbrs[n] # on recupere les voisins du noeuds
		new_cand = cand & nn
		new_done = done & nn 

		# check if we have more to search
		if not new_cand: 
			if not new_done:
				# Found a clique!
				yield clique_so_far[:]
			clique_so_far.pop()
			continue

		# Shortcut--only one node left!
		if not new_done and len(new_cand)==1:
			yield clique_so_far + list(new_cand)
			clique_so_far.pop()
			continue

		# find pivot node (max connected in cand)
		# look in done nodes first
		numb_cand=len(new_cand)
		maxconndone=-1
		for n in new_done:
			cn = new_cand & nnbrs[n]
			conn=len(cn)
			if conn > maxconndone:
				pivotdonenbrs=cn
				maxconndone=conn
				if maxconndone==numb_cand:
					break
			
		# Shortcut--this part of tree already searched
		if maxconndone == numb_cand:  
			clique_so_far.pop()
			continue
			
		# still finding pivot node
		# look in cand nodes second
		maxconn=-1
		for n in new_cand:
			cn = new_cand & nnbrs[n]
			conn=len(cn)
			if conn > maxconn:
				pivotnbrs=cn
				maxconn=conn
				if maxconn == numb_cand-1:
					break 

		# pivot node is max connected in cand from done or cand
		if maxconndone > maxconn:
			pivotnbrs = pivotdonenbrs
		# save search status for later backout
		stack.append( (cand, done, smallcand) )
		cand=new_cand
		done=new_done
		smallcand = cand - pivotnbrs

def compute_cliqueNumber(liste, node):
	cpt = 0
	for clique in liste :
		if node in clique:
			cpt = cpt+1
	return cpt

def compute_RawCliqueScore(liste, node) :
	rawCliqueScore = 0
	for clique in liste:
		if node in clique:
			rawCliqueScore = 2 * math.exp(len(clique)-1) + rawCliqueScore
	return rawCliqueScore

def compute_weightedCliqueScore(liste, node, time) :
	weightedCliqueScore = 0
	for clique in liste:
		if node in clique:
			weightedCliqueScore = time * 2 * math.exp(len(clique)-1) + weightedCliqueScore
	return weightedCliqueScore

# Normalize the metric on mapped to a [0, 100] scale
def NormalizeMetricValue(graph, metric) :
	min = metric.getNodeValue(graph.getOneNode())
	max = metric.getNodeValue(graph.getOneNode())
		
	for n in graph.getNodes():
		valeur = metric.getNodeValue(n)
		if valeur > max:
			max = valeur
		if valeur < min:
			min = valeur	

	for n in graph.getNodes():
		normalizedValue = (100 * ((metric.getNodeValue(n) - min) / (max - min)))
		metric.setNodeValue(n, normalizedValue)
