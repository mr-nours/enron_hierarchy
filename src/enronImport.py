from tulip import *
from email.parser import Parser, HeaderParser
from email.utils import parsedate_tz
from email.utils import mktime_tz
from datetime import datetime
from datetime import timedelta
import os
import math
import collections

# Class used for the mail parsing process. Also, it computes metrics on flux.
# Takes a tlp graph, various tlp properties and a path to the mail directory we need to parse.
class mailParser():
	def __init__(self, graph, receivedMails, sentMails, avgResponseTime, person, filepath):
		self.f = filepath
		self.g = graph
		self.names = self.g.getStringProperty("nodeName")
		self.color = self.g.getColorProperty("viewColor")
		self.receivedMails = receivedMails
		self.sentMails = sentMails
		self.avgResponseTime = avgResponseTime
		self.person = person
		self.users = {}
		self.link = {}
		self.peer = {}
		self.d = timedelta(days = 4)
	
	# Parses a person's sent mails directory
	def parse_sent(self, person):
		# We define a list of paths to parse
		pathList = [self.f + person + "/sent_items/" , self.f + person + "/sent/"]
		expeditors = []	
		# We begin the parsing process for each path in the list	
		for pathToParse in pathList :
			if os.path.exists(pathToParse):
				listing_sent = os.listdir(pathToParse)
				self.users[person]["sent_list"] = []
				# For each mail sent
				for file in listing_sent:
					currentMail = open(pathToParse+file, 'rb')
					msg = HeaderParser().parse(currentMail)
					# We verify if we have all the data and no error
					if type(msg['To']) is str and type(msg['From']) is str and type(msg['Subject']):
						# We don't want to parse a mail corresponding to these special cases
						if "FW:" not in msg['Subject'] and msg['From'] != "no.address@enron.com"  and msg['To'] != "no.address@enron.com" :			
								recipients = msg['To'].split(',')
								date = self.extractDateFromMsg(msg)
								# Preparation for the computation of the metrics on flux
								for to in recipients:
									self.users[person]["sent_list"].append([to.strip(),date])
								expeditor = msg['From']
								expeditors.append(expeditor.strip())
								self.peer[expeditor.strip()] = person
								# Creates a link on the graph between the expeditor and its recipients
								self.register(expeditor, recipients, person)
					currentMail.close()
		# Creates a special link between different adresses used by the same person
		if len(set(expeditors)) > 1:
			self.personWithMultipleAdress(list(set(expeditors)))
	
	# Computes a standard date from a mail date
	def extractDateFromMsg(self, msg):
		date = None
		date_str = msg['Date']
		if(date_str):
			date_tuple = parsedate_tz(date_str)
			if date_tuple:
				date = datetime.fromtimestamp(mktime_tz(date_tuple))
		return date
	
	# Main of our mailParser class
	def parse(self):
		listing = os.listdir(self.f)
		# For each person
		for person in listing:
			self.users[person] = {}
			self.users[person]["response_time"] = 0.0
			self.parse_sent(person)
			self.computeResponseTime(person)
	
	# Computes the average response time score of a person
	def computeResponseTime(self, person):
		delta_cpt = 0
		guyMails = []
		# Determines the guy's "person" mails.
		for mail in self.peer:
			if self.peer[mail] == person:
				guyMails.append(mail)
		# For each mail sent, we check if there is a response to one of guy's "person" mails.
		for pair in self.users[person]["sent_list"]:
			# If the destination mail has an enron employee associated to itself.
			if self.peer.has_key(pair[0]):
				for p in self.users[self.peer[pair[0]]]["sent_list"]:
					# Response may be found
					if p[0] in guyMails:
						if p[1] > pair[1]:
							delta = p[1] - pair[1]
							# If the response is not too old, we take it into account
							if delta <= self.d:
								delta_cpt = delta_cpt + 1
								self.users[person]["response_time"] = self.users[person]["response_time"] + delta.total_seconds()
		# We then register the score into the corresponding tlp property
		if delta_cpt > 0:
			self.users[person]["response_time"] = self.users[person]["response_time"] / delta_cpt
			for adr in guyMails:
				for node in self.g.getNodes():
					if adr == self.names[node]:
						self.avgResponseTime[node] = 1/self.users[person]["response_time"]
	
	# Creates a special link between the different adresses used by a person
	def personWithMultipleAdress(self, otherAdresses):
		first = otherAdresses.pop()
		for other in otherAdresses:
			for node in self.g.getNodes():
				if other == self.names[node]:
					otherNode = node
				if first == self.names[node]:
					firstNode = node
			# Add an edge between the similar email adresses 
			edge = self.g.addEdge(otherNode,firstNode)
			self.color[edge] = tlp.Color(255, 0, 0)
	
	# Creates the links between the expeditor and its recipients 
	def register(self, expeditor, recipients, person):
		# Expeditor computing
		flag = False
		expeditor = expeditor.strip()
		for nodes in self.g.getNodes():
			if expeditor == self.names[nodes]:
				flag = True
				expeditorNode = nodes
				break
		# If the expeditor is not in the graph, we create a new node
		if not flag:
			node = self.g.addNode()
			expeditorNode = node
			self.names[node] = expeditor

		# Preprocessing of the metrics on flux
		self.person[expeditorNode] = person
		self.link[person] = expeditorNode
		self.sentMails[expeditorNode] = self.sentMails[expeditorNode] + len(recipients)  

		# Recipients computing
		for adress in recipients:
			adress = adress.strip()
			# We don't want to take into account a mail sent from oneself to oneself
			if expeditor != adress:			
				flag = False
				for node in self.g.getNodes():
					if adress == self.names[node]:
						flag = True
						receptorNode = node
						break
				# If the recipient is not in the graph, we create a new node
				if not flag:
					node = self.g.addNode()
					receptorNode = node
					self.names[node] = adress
				# We create the connexion between the expeditor and the recipients
				edge = self.g.existEdge(expeditorNode, receptorNode)
				if not edge.isValid():
					edge = self.g.addEdge(expeditorNode, receptorNode)
				# Needed for the final visualisation 	
				edgeNumber = self.g.getDoubleProperty("mailNumber")			
				edgeNumber[edge] = edgeNumber[edge] + 1
				# We increment the number of received mails of the recipient
				self.receivedMails[receptorNode] = self.receivedMails[receptorNode] + 1
				
	
	# Deprecated. Used to transfer a class property into a Tlp property
	def propertiesToTulipProperties(self):
		for person, node in self.link.items():
			self.avgResponseTime[node] = self.users[person]["response_time"]


#End of MailParse class
 
def find_cliques(graph):
	"""Search for all maximal cliques in a graph.

    Maximal cliques are the largest complete subgraph containing
    a given node.  The largest maximal clique is sometimes called
    the maximum clique.

    Returns
    -------
    generator of lists: generator of member list for each maximal clique
    
    Notes
    -----
    To obtain a list of cliques, use list(find_cliques(G)).

    Based on the algorithm published by Bron & Kerbosch (1973)
    as adapated by Tomita, Tanaka and Takahashi (2006)
    and discussed in Cazals and Karande (2008).
    
    This algorithm ignores self-loops and parallel edges as
    clique is not conventionally defined with such edges.

    There are often many cliques in graphs.  This algorithm can
    run out of memory for large graphs.
    
     References
    ----------
    .. [1] Bron, C. and Kerbosch, J. 1973.
       Algorithm 457: finding all cliques of an undirected graph.
       Commun. ACM 16, 9 (Sep. 1973), 575-577.
       http://portal.acm.org/citation.cfm?doid=362342.362367

    .. [2] Etsuji Tomita, Akira Tanaka, Haruhisa Takahashi,
       The worst-case time complexity for generating all maximal
       cliques and computational experiments,
       Theoretical Computer Science, Volume 363, Issue 1,
       Computing and Combinatorics,
       10th Annual International Conference on
       Computing and Combinatorics (COCOON 2004), 25 October 2006, Pages 28-42
       http://dx.doi.org/10.1016/j.tcs.2006.06.015

    .. [3] F. Cazals, C. Karande,
       A note on the problem of reporting maximal cliques,
       Theoretical Computer Science,
       Volume 407, Issues 1-3, 6 November 2008, Pages 564-568,
       http://dx.doi.org/10.1016/j.tcs.2008.05.010
    """
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


# Computes the authority and the hub score for each node in the input graph.
# Adaptation of the HITS algorithm by Jon Kleinberg.
# Defines a maximum number of iterations, and a tolerance value in order to verify
# that auth and hub do converge.
def hits(graph, auth, hub, max_iter=100, tol=1.0e-8):
        i = 0
        # Initialization step
        for n in graph.getNodes():
                auth[n] = hub[n] = 1.0 / graph.numberOfNodes()
        # Perpetual loop. Stops when max_iter is reached or when hub converges.
        while True:
                hublast = hub
                norm = 1.0
                normsum = 0
                err = 0
                # We update the auth score for each node
                for n in graph.getNodes():
                        auth[n] = 0
                        for v in graph.getInNodes(n):
                                auth[n] = auth[n] + hub[v]
                        normsum = normsum + auth[n]
                norm = norm / normsum
                # Normalization of auth
                for n in graph.getNodes():
                        auth[n] = auth[n] * norm
                norm = 1.0
                # We update the hub score for each node
                for n in graph.getNodes():
                        hub[n] = 0
                        for v in graph.getOutNodes(n):
                                hub[n] = hub[n] + auth[v]
                        normsum = normsum + hub[n]
                norm = norm / normsum
                # Normalization of hub
                for n in graph.getNodes():
                        hub[n] = hub[n] * norm
                for n in graph.getNodes():
                        err = err + abs(hub[n] - hublast[n])
                # When hub converges, we can stop the perpetual loop
                if err < tol:
                        break
                # But if we reach max_iter, then it means we failed the calculation
                if i > max_iter:
                        raise HITSError(\
                        "HITS: failed to converge in %d iterations."%(i+1))
                i = i + 1
        return auth,hub
	
def compute_mailNumber(node,receivedMails,sentMails,totalMail):
        totalMail[node] = receivedMails[node] + sentMails[node]

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

# Normalizes the metrics, mapped to a [0, 100] scale
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

# Normalizes the metrics, mapped to a [0, 100] scale
# Edges version
def NormalizeEdgeMetricValue(graph, metric) :
	min = metric.getNodeValue(graph.getOneNode())
	max = metric.getNodeValue(graph.getOneNode())
		
	for n in graph.getEdges():
		valeur = metric.getEdgeValue(n)
		if valeur > max:
			max = valeur
		if valeur < min:
			min = valeur	

	for n in graph.getEdges():
		normalizedValue = (100 * ((metric.getEdgeValue(n) - min) / (max - min)))
		metric.setEdgeValue(n, normalizedValue)

# We use a set of normalized metrics to compute the social score.
# metrics is a list of the metrics we want to use to compute the social score.
# storageProperty is where we put the social score.
def compute_SocialScore(graph, metrics, storageProperty) :
	for n in graph.getNodes():
		totalWeightedContribution = 0	
		totalWeigth = 0	
		for metric,weight in metrics.items():
			totalWeightedContribution = totalWeightedContribution + metric.getNodeValue(n) * weight
			totalWeigth = totalWeigth + weight		
		score = totalWeightedContribution/totalWeigth
		storageProperty.setNodeValue(n,score)

# Merges nodes representing different adresses of a person.
def node_Fusion(graph,color,receivedMails,sentMails,avgResponseTime,nodeName,tolerance):
	nodesTodelete = []
	# If an adress is used by more than "tolerance" user, it is considered a common shared adress	
	for n in graph.getNodes():
		delta = 0
		time = 0
		multi = False
		# We verify if the current node is not a shared email account	
		redNumberNode = 0		
		for vedge in graph.getOutEdges(n):
			if color[vedge] == tlp.Color(255, 0, 0):
				redNumberNode = redNumberNode + 1	
		if redNumberNode <= tolerance:	
			# Beginning of the merge			
			for edge in graph.getInEdges(n):
				if color[edge] ==tlp.Color(255, 0, 0):
					sourceEdge = graph.source(edge)
					# We verify if the distant email account that we want to merge is not a common shared adress					
					redNumber = 0					
					for e in graph.getOutEdges(sourceEdge):
						if color[e] == tlp.Color(255, 0, 0):
							redNumber = redNumber + 1
					if redNumber <= tolerance:  
						# Merging process						
						multi = True	
						# Target redirection
						edgesSourceInEdge = graph.getInEdges(sourceEdge)				
						for edges in edgesSourceInEdge:				 												
							# If the edge is red, we update the target							
							if color[edges] == tlp.Color(255, 0, 0):
								graph.setTarget(edges,n) 							
							else :							
								# If edge doesn't already exist
								edgeExist = graph.existEdge(graph.source(edges), n, False)			
								if not edgeExist.isValid():	
									graph.setTarget(edges,n)				
						# Source redirection
						edgesSourceOutEdge = graph.getOutEdges(sourceEdge)
						for edges in edgesSourceOutEdge:
							# If the edge is red, we update the source							
							if color[edges] == tlp.Color(255, 0, 0):
								graph.setSource(edges,n)
							else :
								# If edge doesn't already exist
								target = graph.target(edges)
								if target != n:					
									edgeExist = graph.existEdge(n, target, False)	
									if not edgeExist.isValid():			
										graph.setSource(edges,n)	
						# Delete the current red edge				
						graph.delEdge(edge)
						nodesTodelete.append(sourceEdge)	 
						# Values merging
						receivedMails[n] = receivedMails[n] + receivedMails[sourceEdge]
						sentMails[n] = sentMails[n] + sentMails[sourceEdge]
						nodeName[n] = nodeName[n]+";"+nodeName[sourceEdge]
						time = time + avgResponseTime[sourceEdge]
						delta = delta + 1
				if multi == True :
					avgResponseTime[n] = (avgResponseTime[n] + time) / (delta+1)		
					
	# Deletion of the merged nodes			
	for node in list(set(nodesTodelete)):
		graph.delNode(node)	

# Utility algorithm to ease the computation of the hierarchical view
# It sorts the tlp property and returns a python sorted list
def build_OrderedList(graph, socialScore):
	tuple = collections.namedtuple('tuple', 'score name')
	list = {}
	
	for n in graph.getNodes():
		list[n] = socialScore[n]

	best = sorted([tuple(v,k) for (k,v) in list.items()], reverse=True)
	return best

# Builds the hierarchical view
def build_SocialHierarchy(graph, nodeList, stepList, height, width):
	viewLayout =  graph.getLayoutProperty("viewLayout")
	socialScore = graph.getDoubleProperty("socialScore")
	viewSize =  graph.getSizeProperty("viewSize")
	nodeName = graph.getStringProperty("nodeName")
	viewLabel =  graph.getStringProperty("viewLabel")
	person = graph.getStringProperty("person")

	currentStep = stepList.pop()	
	positionY = 0
	i = 0	
	toDo = []

	
	while i <= graph.numberOfNodes()-1 :	
		node = nodeList[i].name
		if socialScore[node] >= currentStep:		
			toDo.append(node)
			i = i+1

		if i > graph.numberOfNodes()-1 or socialScore[node] < currentStep :
			if len(toDo) > 0 :		
				positionY = positionY - height				
				# Process the list
				firstNode = toDo.pop(0)	
				nodeSize = viewSize[firstNode].getW()	
				leftPositionX = ((nodeSize+width) - ((len(toDo)+1) * (nodeSize+width)))/2
				coord = tlp.Coord()
				coord.setX(leftPositionX)
				coord.setY(positionY)
				viewLayout[firstNode] = coord
				viewLabel[firstNode] = person[firstNode]
				
				for n in toDo:					
					leftPositionX = leftPositionX + nodeSize + width
					coord.setX(leftPositionX)
					coord.setY(positionY)
					viewLayout[n] = coord
					viewLabel[n] = person[n]
					nodeSize = viewSize[n].getW()			
				toDo = []
			if len(stepList) > 0 : 	
				currentStep = stepList.pop()
	
# Deletes negligible nodes
def threshold(graph, value):
	receivedMails = graph.getDoubleProperty("received_mails")
	sentMails = graph.getDoubleProperty("sent_mails")
	totalMail = graph.getDoubleProperty("totalMail")	

	for n in graph.getNodes():
		if totalMail[n] < 20:
			graph.delNode(n)

# Deletes non-Enron nodes
def enronFilter(graph,person):
	for n in graph.getNodes():
		if person[n] == "" :
			graph.delNode(n)

def visualisationCustom(graph):
	hub = graph.getDoubleProperty("hub")
	nodeDegree =  graph.getDoubleProperty("nodeDegree")
	cliqueNumber =  graph.getDoubleProperty("cliqueNumber")
	color = graph.getColorProperty("viewColor")
	viewLabel =  graph.getStringProperty("viewLabel")
	person = graph.getStringProperty("person")
	viewSize =  graph.getSizeProperty("viewSize")
	viewBorderColor =  graph.getColorProperty("viewBorderColor")
	socialScore = graph.getDoubleProperty("socialScore")
	viewFontSize =  graph.getIntegerProperty("viewFontSize")
	viewLabelPosition =  graph.getIntegerProperty("viewLabelPosition")
	viewLabel =  graph.getStringProperty("viewLabel")
	person = graph.getStringProperty("person")
	viewBorderColor =  graph.getColorProperty("viewBorderColor")
	viewBorderWidth =  graph.getDoubleProperty("viewBorderWidth")
	mailNumber = graph.getDoubleProperty("mailNumber")

	for n in graph.getNodes():
		text = int(hub[n]/2)	
		viewFontSize[n] = 25	
		viewLabelPosition[n] = tlp.LabelPosition.Top

		s = int(hub[n]/10)	
		if s == 0 :
			s = 1	
		viewSize[n] = tlp.Size(s,s,s)
		sc = int(socialScore[n])
		nd = int(nodeDegree[n]*2.5)
		if sc >= 57 : 		
			color[n] = tlp.Color(205,42,42,nd)
			viewBorderColor[n] = tlp.Color(205,42,42,150) 
		elif sc >= 44 :
			color[n] = tlp.Color(250,137,17,nd)
			viewBorderColor[n] = tlp.Color(250,137,17,150)
		elif sc >= 32 :
			color[n] = tlp.Color(105,138,67,nd)
			viewBorderColor[n] = tlp.Color(105,138,67,150)
		elif sc >= 21 : 
			color[n] = tlp.Color(125,165,211,nd)
			viewBorderColor[n] = tlp.Color(121,165,211,150)
		else :
			color[n] = tlp.Color(131,131,131,nd)
			viewBorderColor[n] = tlp.Color(224,224,224,150)
	
		viewBorderWidth[n] = 1	
	
	for edge in graph.getEdges():
		s = mailNumber[edge]
		couleur = color[graph.source(edge)]
		couleur.setA(int((s+20)*2))
		color[edge] = couleur
		viewBorderColor[edge] = tlp.Color(0,0,0,0)

def main(graph): 
	for edge in graph.getEdges():
		graph.delEdge(edge)
	for node in graph.getNodes():
		graph.delNode(node)

	#Properties of the graph
	nodeName = graph.getStringProperty("nodeName")
	nodeDegree =  graph.getDoubleProperty("nodeDegree")
	betweenessCentrality =  graph.getDoubleProperty("betweenessCentrality")
	cliqueNumber =  graph.getDoubleProperty("cliqueNumber")	
	rawUserCliqueScore = graph.getDoubleProperty("rawUserCliqueScore")
	clusteringCoefficient = graph.getDoubleProperty("clusteringCoefficient")
	shortestPath = graph.getDoubleProperty("shortestPath")
	socialScore = graph.getDoubleProperty("socialScore")
	auth = graph.getDoubleProperty("auth")
	hub = graph.getDoubleProperty("hub")
	color = graph.getColorProperty("viewColor")
	viewLabel =  graph.getStringProperty("viewLabel")
	receivedMails = graph.getDoubleProperty("received_mails")
	sentMails = graph.getDoubleProperty("sent_mails")
	totalMail = graph.getDoubleProperty("totalMail")
	avgResponseTime = graph.getDoubleProperty("response_time")
	weightedCliqueScore = graph.getDoubleProperty("weightedCliqueScore")
	person = graph.getStringProperty("person")
	mailNumber = graph.getDoubleProperty("mailNumber")
	
	# Email Corpus parsing
	#enronpath = "C:/Users/admin/Downloads/enron_mail_20110402/"
	enronpath = "C:/Users/samuel/Desktop/ENRON/enron_mail_20110402/maildir/"
	myParser = mailParser(graph, receivedMails, sentMails, avgResponseTime, person, enronpath)
	myParser.parse()

	# Merging of multimail
	node_Fusion(graph,color,receivedMails,sentMails,avgResponseTime, nodeName, 1)	

	# Cleans the graph : deletion of non signifiant nodes
	#threshold(graph, 20)

	# Structural indicators
	liste = list(find_cliques(graph))
	for n in graph.getNodes():	
		compute_mailNumber(n,receivedMails,sentMails,totalMail)
		nbOccurence = compute_cliqueNumber(liste,n)
		cliqueNumber.setNodeValue(n,nbOccurence)
		CliqueScore = compute_RawCliqueScore(liste,n)
		rawUserCliqueScore.setNodeValue(n, CliqueScore)
		weightedScore = compute_weightedCliqueScore(liste,n,avgResponseTime.getNodeValue(n))
		weightedCliqueScore.setNodeValue(n,weightedScore)
	
	# Degree centrality (nodeDegree)
	graph.computeDoubleProperty("Degree", nodeDegree)

	# Clustering coefficient (clusteringCoefficient)
	tlp.clusteringCoefficient(graph,clusteringCoefficient)

	# Mean of shortest path lenght (shortestPath)
	dataSet = tlp.DataSet()
	dataSet["closeness centrality"] = True
	dataSet["norm"] = True
	graph.computeDoubleProperty("Eccentricity", shortestPath, dataSet)
	
	# Betweeness centrality (betweenessCentrality)
	graph.computeDoubleProperty("Betweenness Centrality", betweenessCentrality)

	# Hub and Authorities importance
	hits(graph, auth, hub)

	# Enron filter : deletes non enron persons
	enronFilter(graph, person)
	
	# Normalisation des indicateurs sur une echelle [0-100]
	NormalizeMetricValue(graph,totalMail)
	NormalizeMetricValue(graph,avgResponseTime)
	NormalizeMetricValue(graph,cliqueNumber)
	NormalizeMetricValue(graph,rawUserCliqueScore)
	NormalizeMetricValue(graph,weightedCliqueScore)
	NormalizeMetricValue(graph,nodeDegree)
	NormalizeMetricValue(graph,clusteringCoefficient)
	NormalizeMetricValue(graph,shortestPath)
	NormalizeMetricValue(graph,betweenessCentrality)
	NormalizeMetricValue(graph,hub)
	NormalizeMetricValue(graph,auth)
	# Edges : only for visualisation	
	NormalizeEdgeMetricValue(graph,mailNumber)
	
	# List of metrics with their weight to compute the social score
	metrics = {totalMail:0.5, cliqueNumber:2, rawUserCliqueScore:4, nodeDegree:2, clusteringCoefficient:1,shortestPath:5, betweenessCentrality:2, hub:4, auth:1}
	compute_SocialScore(graph,metrics,socialScore)

	# Display Hierarchy
	orderedList = build_OrderedList(graph, socialScore)
	
	stepList = [0,21, 32,44, 57,84]	
	visualisationCustom(graph)	
	build_SocialHierarchy(graph, orderedList, stepList, 10,5)
	
