from tulip import *
from email.parser import Parser
from email.utils import parsedate_tz
from email.utils import mktime_tz
from datetime import datetime
from datetime import timedelta
from moduleclique import find_cliques
import os

class mailParser():
	def __init__(self, graph, nodeName, filepath):
		self.f = filepath
		self.g = graph
		self.names = nodeName
		self.color = self.g.getColorProperty("viewColor")
		self.users = {}
		self.d = timedelta(days = 4)
	
	def parse_sent(self, person):
		pathToParse = self.f + person + "/sent_items/"
		listing_sent = os.listdir(pathToParse)
		expeditors = []
		sent_mails = 0
		self.users[person]["sent_list"] = []
		# For each mail sent
		for file in listing_sent:
			sent_mails = sent_mails + 1
   			currentMail = open(pathToParse+file, 'rb')
   			msg = Parser().parse(currentMail)
   			if type(msg['To']) is str:
   				recipients = msg['To'].split(',')
   				date = self.extractDateFromMsg(msg)
   				for to in recipients:
   					self.users[person]["sent_list"].append([to.strip(),date])
   			if type(msg['From']) is str:
   				expeditor = msg['From']
   				expeditors.append(expeditor)
   			self.register(expeditor, recipients)
   			currentMail.close()
   		self.users[person]["sent_mails"] = sent_mails
   		self.personWithMultipleAdress(list(set(expeditors)))
   		
   	def parse_received(self, person):
   	   pathToParse = self.f + person + "/inbox/"
   	   listing_received = os.listdir(pathToParse)
   	   received_mails = 0
   	   delta_cpt = 0
   	   for file in listing_received:
   			if os.path.isfile(pathToParse+file):
   				received_mails = received_mails + 1
	   			currentMail = open(pathToParse+file, 'rb')
	   			msg = Parser().parse(currentMail)
	   			if type(msg['From']) is str:
	   				expeditor = msg['From'].strip()
	   				date = self.extractDateFromMsg(msg)
	   				for pair in self.users[person]["sent_list"]:
	   					if expeditor == pair[0]:
	   						if date > pair[1]:
	   							delta = date - pair[1]
	   							if delta <= self.d:
	   								delta_cpt = delta_cpt + 1
	   								self.users[person]["response_time"] = self.users[person]["response_time"] + delta.total_seconds()
	   								break
	   self.users[person]["received_mails"] = received_mails
	   self.users[person]["response_time"] = self.users[person]["response_time"] / delta_cpt
   		#print(received_mails)
	
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
			self.parse_received(person)
	   #	print self.users
   	
	def personWithMultipleAdress(self, otherAdresses):
		first = otherAdresses.pop()
		for other in otherAdresses:
			for node in self.g.getNodes():
				if other == self.names[node]:
					otherNode = node
				if first == self.names[node]:
					firstNode = node
			edge = self.g.addEdge(firstNode, otherNode)
			self.color[edge] = tlp.Color(255, 0, 0)
   	
   	def register(self, expeditor, recipients):
   		#Traitement de l'expediteur
   		flag = False
   		for nodes in self.g.getNodes():
   			if expeditor in self.names[nodes]:
   				flag = True
   				expeditorNode = nodes
   				break
   		if not flag:
   			node = self.g.addNode()
   			expeditorNode = node
   			self.names[node] = expeditor.strip()
   		#Traitement des recepteurs
   		for adress in recipients:
   			adress = adress.strip()
   			flag = False
   			for node in self.g.getNodes():
   				if adress in self.names[node]:
   					flag = True
   					receptorNode = node
   					break
   			if not flag:
   				node = self.g.addNode()
   				receptorNode = node
   				self.names[node] = adress
   			self.g.addEdge(expeditorNode, receptorNode)

def find_cliques(G):
    """
    Search for all maximal cliques in a graph.
 
    This algorithm searches for maximal cliques in a graph.
    maximal cliques are the largest complete subgraph containing
    a given point.  The largest maximal clique is sometimes called
    the maximum clique.
 
    This implementation is a generator of lists each
    of which contains the members of a maximal clique.
    To obtain a list of cliques, use list(find_cliques(G)).
    The method essentially unrolls the recursion used in
    the references to avoid issues of recursion stack depth.

    Notes
    -----
    Based on the algorithm published by Bron & Kerbosch (1973) [1]_
    as adapated by Tomita, Tanaka and Takahashi (2006) [2]_
    and discussed in Cazals and Karande (2008) [3]_.

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

    # Cache nbrs and find first pivot (highest degree)
    maxconn=-1
    nnbrs={}
    pivotnbrs=set() # handle empty graph
    for n,nbrs in G.adjacency_iter():
        conn = len(nbrs)
        if conn > maxconn:
            nnbrs[n] = pivotnbrs = set(nbrs)
            maxconn = conn
        else:
            nnbrs[n] = set(nbrs)
    # Initial setup
    cand=set(nnbrs)
    smallcand = cand - pivotnbrs
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
        cand.remove(n)
        done.add(n)
        nn=nnbrs[n]
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

def main(graph): 
	for edge in graph.getEdges():
		graph.delEdge(edge)
	for node in graph.getNodes():
		graph.delNode(node)
	nodeName =  graph.getStringProperty("nodeName")
	myParser = mailParser(graph, nodeName, "/net/cremi/cbadiola/travail/bioInfo/enron_mail_20110402/maildirtest/")
	myParser.parse()
