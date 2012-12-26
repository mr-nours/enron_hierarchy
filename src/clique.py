from tulip import *

def main(graph): 
	liste =  list(find_cliques(graph))
	cliqueNumber =  graph.getIntegerProperty("cliqueNumber")
	rawCliqueScore = graph.getDoubleProperty("rawCliqueScore")
	for n in graph.getNodes():	
		nbOccurence = compute_cliqueNumber(liste,n)
		cliqueNumber.setNodeValue(n,nbOccurence)

	ClusteringCoefficient = averageClusteringCoefficient(graph)
	graph.applyAlgorithm('Betweeness Centrality')
	print ClusteringCoefficient

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
