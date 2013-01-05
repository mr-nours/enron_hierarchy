from tulip import *
from email.parser import Parser, HeaderParser
from email.utils import parsedate_tz
from email.utils import mktime_tz
from datetime import datetime
from datetime import timedelta
import os
import math
import collections

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
	
	def parse_sent(self, person):
		pathToParse = [self.f + person + "/sent_items/" , self.f + person + "/sent/"]
		expeditors = []		
		for path in pathToParse :
			if os.path.exists(pathToParse):
				#pathToParse = self.f + person + "/sent/"
				listing_sent = os.listdir(pathToParse)
				self.users[person]["sent_list"] = []
				# For each mail sent
				for file in listing_sent:
					currentMail = open(pathToParse+file, 'rb')
					msg = HeaderParser().parse(currentMail)
					# We verify if we have all the data
					if type(msg['To']) is str and type(msg['From']) is str and type(msg['Subject']):
						if "FW:" not in msg['Subject'] and msg['From'] != "no.address@enron.com"  and msg['To'] != "no.address@enron.com" :			
								recipients = msg['To'].split(',')
								date = self.extractDateFromMsg(msg)
								for to in recipients:
									self.users[person]["sent_list"].append([to.strip(),date])
								expeditor = msg['From']
								expeditors.append(expeditor.strip())
								self.peer[expeditor.strip()] = person
								self.register(expeditor, recipients, person)
					currentMail.close()
		if len(set(expeditors)) > 1:
			self.personWithMultipleAdress(list(set(expeditors)))
	
	def extractDateFromMsg(self, msg):
		date = None
		date_str = msg['Date']
		if(date_str):
			date_tuple = parsedate_tz(date_str)
			if date_tuple:
				date = datetime.fromtimestamp(mktime_tz(date_tuple))
		return date
	
	def parse(self):
		listing = os.listdir(self.f)
		# For each person
		for person in listing:
			self.users[person] = {}
			self.users[person]["response_time"] = 0.0
			self.parse_sent(person)
			#self.parse_received(person)
			self.computeResponseTime(person)
		#self.propertiesToTulipProperties()
		#    print self.users
	
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
							if delta <= self.d:
								delta_cpt = delta_cpt + 1
								self.users[person]["response_time"] = self.users[person]["response_time"] + delta.total_seconds()
		if delta_cpt > 0:
			self.users[person]["response_time"] = self.users[person]["response_time"] / delta_cpt
			for adr in guyMails:
				for node in self.g.getNodes():
					if adr == self.names[node]:
						self.avgResponseTime[node] = self.users[person]["response_time"]
	
	def personWithMultipleAdress(self, otherAdresses):
		first = otherAdresses.pop()
		for other in otherAdresses:
			for node in self.g.getNodes():
				if other == self.names[node]:
					otherNode = node
				if first == self.names[node]:
					firstNode = node
			# Add an edge between the similar email adress 
			edge = self.g.addEdge(otherNode,firstNode)
			self.color[edge] = tlp.Color(255, 0, 0)
	
	def register(self, expeditor, recipients, person):
		# Traitement de l'expediteur
		flag = False
		expeditor = expeditor.strip()
		for nodes in self.g.getNodes():
			if expeditor == self.names[nodes]:
				flag = True
				expeditorNode = nodes
				break

		if not flag:
			node = self.g.addNode()
			expeditorNode = node
			self.names[node] = expeditor

		self.person[expeditorNode] = person
		self.link[person] = expeditorNode
		self.sentMails[expeditorNode] = self.sentMails[expeditorNode] + 1  

		#Traitement des recepteurs
		for adress in recipients:
			adress = adress.strip()
			if expeditor != adress:			
				flag = False
				for node in self.g.getNodes():
					if adress == self.names[node]:
						flag = True
						receptorNode = node
						break
				if not flag:
					node = self.g.addNode()
					receptorNode = node
					self.names[node] = adress
				edge = self.g.existEdge(expeditorNode, receptorNode)
				if not edge.isValid():
					self.g.addEdge(expeditorNode, receptorNode)
	
				# We increment the number of received Mail of the recipient
				self.receivedMails[receptorNode] = self.receivedMails[receptorNode] + 1
				
	
	
	def propertiesToTulipProperties(self):
		for person, node in self.link.items():
			self.avgResponseTime[node] = self.users[person]["response_time"]


#End of MailParse class

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


    
def hits(graph, auth, hub, max_iter=100, tol=1.0e-8):
        i = 0
        for n in graph.getNodes():
                auth[n] = hub[n] = 1.0 / graph.numberOfNodes()
        while True:
                hublast = hub
                norm = 1.0
                normsum = 0
                err = 0
                for n in graph.getNodes():
                        auth[n] = 0
                        for v in graph.getInNodes(n):
                                auth[n] = auth[n] + hub[v]
                        normsum = normsum + auth[n]
                norm = norm / normsum
                for n in graph.getNodes():
                        auth[n] = auth[n] * norm
                norm = 1.0
                for n in graph.getNodes():
                        hub[n] = 0
                        for v in graph.getOutNodes(n):
                                hub[n] = hub[n] + auth[v]
                        normsum = normsum + hub[n]
                norm = norm / normsum
                for n in graph.getNodes():
                        hub[n] = hub[n] * norm
                for n in graph.getNodes():
                        err = err + abs(hub[n] - hublast[n])
                if err < tol:
                        break
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

def compute_SocialScore(graph, metrics, storageProperty) :
	for n in graph.getNodes():
		totalWeightedContribution = 0	
		totalWeigth = 0	
		for metric,weight in metrics.items():
			totalWeightedContribution = totalWeightedContribution + metric.getNodeValue(n) * weight
			totalWeigth = totalWeigth + weight		
		score = totalWeightedContribution/totalWeigth
		storageProperty.setNodeValue(n,score)

def node_Fusion(graph,color,receivedMails,sentMails,avgResponseTime,nodeName,tolerance):
	nodesTodelete = []
	# If an adress is used by more of tolerance user, it is considered as a common adress email	
	for n in graph.getNodes():
		delta = 0
		time = 0
		multi = False
		# We verify if the current node is not a partaged email account		
		redNumberNode = 0		
		for vedge in graph.getOutEdges(n):
			if color[vedge] == tlp.Color(255, 0, 0):
				redNumberNode = redNumberNode + 1	
		if redNumberNode <= tolerance:	
			# Beginin of the fusion			
			for edge in graph.getInEdges(n):
				if color[edge] ==tlp.Color(255, 0, 0):
					sourceEdge = graph.source(edge)
					# We verify if the distant email account that we want fusion is not a common email adress					
					redNumber = 0					
					for e in graph.getOutEdges(sourceEdge):
						if color[e] == tlp.Color(255, 0, 0):
							redNumber = redNumber + 1
					if redNumber <= tolerance:  
						# Fusion process						
						multi = True	
						# Target redirection
						edgesSourceInEdge = graph.getInEdges(sourceEdge)				
						for edges in edgesSourceInEdge:				 												
							# If the edge is red, we update the target							
							if color[edges] == tlp.Color(255, 0, 0):
								graph.setTarget(edges,n) 							
							else :							
								# If edge don't already exist
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
								# If edge don't already exist
								target = graph.target(edges)
								if target != n:					
									edgeExist = graph.existEdge(n, target, False)	
									if not edgeExist.isValid():			
										graph.setSource(edges,n)	
						# Delete the current red Edge				
						graph.delEdge(edge)
						nodesTodelete.append(sourceEdge)	 
						# Values fusion
						receivedMails[n] = receivedMails[n] + receivedMails[sourceEdge]
						sentMails[n] = sentMails[n] + sentMails[sourceEdge]
						nodeName[n] = nodeName[n]+";"+nodeName[sourceEdge]
						time = time + avgResponseTime[sourceEdge]
						delta = delta + 1
				if multi == True :
					avgResponseTime[n] = (avgResponseTime[n] + time) / (delta+1)		
					
	# Deletion of the nodes fusioned			
	for node in list(set(nodesTodelete)):
		graph.delNode(node)	

def build_OrderedList(graph, socialScore):
	tuple = collections.namedtuple('tuple', 'score name')
	list = {}
	
	for n in graph.getNodes():
		list[n] = socialScore[n]

	best = sorted([tuple(v,k) for (k,v) in list.items()], reverse=True)
	return best
	
def build_SocialHierarchy(graph, nodeList, level, height, width):
	viewLayout =  graph.getLayoutProperty("viewLayout")
	socialScore = graph.getDoubleProperty("socialScore")
	viewSize =  graph.getSizeProperty("viewSize")
	nodeName = graph.getStringProperty("nodeName")
	viewLabel =  graph.getStringProperty("viewLabel")

	step = 100/level
	currentStep = 100 - step
	positionY = 0
	i = 1	
	toDo = []

	# First node position
	node = nodeList[0].name
	coord = tlp.Coord(0,0,0)
	viewLayout[node] = coord
	viewLabel[node] = nodeName[node].split(';')[0]

	
	while i <= graph.numberOfNodes()-1 :	
		node = nodeList[i].name
		if socialScore[node] >= currentStep:		
			toDo.append(node)
			i = i+1

		if i > graph.numberOfNodes()-1 or socialScore[node] < currentStep :
			if len(toDo) > 0 :		
				positionY = positionY - height				
				# Traiter la liste
				firstNode = toDo.pop(0)	
				nodeSize = viewSize[firstNode].getW()	
				leftPositionX = ((nodeSize+width) - ((len(toDo)+1) * (nodeSize+width)))/2
				coord = tlp.Coord()
				coord.setX(leftPositionX)
				coord.setY(positionY)
				viewLayout[firstNode] = coord
				viewLabel[firstNode] = nodeName[firstNode].split(';')[0]
				
				for n in toDo:					
					leftPositionX = leftPositionX + nodeSize + width
					coord.setX(leftPositionX)
					coord.setY(positionY)
					viewLayout[n] = coord
					viewLabel[n] = nodeName[n].split(';')[0]
					nodeSize = viewSize[n].getW()			
				toDo = []
				
			currentStep = currentStep-step		

def threshold(graph):
	receivedMails = graph.getDoubleProperty("received_mails")
	sentMails = graph.getDoubleProperty("sent_mails")	

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
	
	# Email Corpus parsing
	#enronpath = "C:/Users/admin/Downloads/enron_mail_20110402/"
	enronpath = "C:/Users/samuel/Desktop/ENRON/enron_mail_20110402/maildir/"
	myParser = mailParser(graph, receivedMails, sentMails, avgResponseTime, person, enronpath)
	myParser.parse()

	# Fusion of multimail
	node_Fusion(graph,color,receivedMails,sentMails,avgResponseTime, nodeName, 1)	


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
	tlp.clusteringCoefficient(graph,clusteringCoefficient,graph.numberOfNodes())

	# Mean of shortest path lenght (shortestPath)
	dataSet = tlp.DataSet()
	dataSet["closeness centrality"] = True
	dataSet["norm"] = False
	graph.computeDoubleProperty("Eccentricity", shortestPath, dataSet)
	
	# Betweeness centrality (betweenessCentrality)
	graph.computeDoubleProperty("Betweenness Centrality", betweenessCentrality)
	
	# Hub and Authorities importance
	hits(graph, auth, hub)

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
	
	# List of metrics with their weight to compute the social score
	metrics = {totalMail:1, avgResponseTime:1, cliqueNumber:1, rawUserCliqueScore:1,weightedCliqueScore:1, nodeDegree:1, clusteringCoefficient:1,shortestPath:1, betweenessCentrality:1, hub:1, auth:1}
	compute_SocialScore(graph,metrics,socialScore)

	# Display Hierarchy
	orderedList = build_OrderedList(graph, socialScore)
	build_SocialHierarchy(graph, orderedList, 10, 10,5)
