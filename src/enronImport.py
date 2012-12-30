from tulip import *
from email.parser import Parser
from email.utils import parsedate_tz
from email.utils import mktime_tz
from datetime import datetime
from datetime import timedelta
import os
import math

class mailParser():
	def __init__(self, graph, nodeName, filepath):
		self.f = filepath
		self.g = graph
		self.names = nodeName
		self.color = self.g.getColorProperty("viewColor")
		self.receivedMails = self.g.getIntegerProperty("received_mails")
		self.sentMails = self.g.getIntegerProperty("sent_mails")
		self.avgResponseTime = self.g.getDoubleProperty("response_time")
		self.users = {}
		self.link = {}
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
			self.register(expeditor, recipients, person)
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
			self.propertiesToTulipProperties()
		#    print self.users
	
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
	
	def register(self, expeditor, recipients, person):
		#Traitement de l'expediteur
		flag = False
		expeditor = expeditor.strip()
		for nodes in self.g.getNodes():
			if expeditor in self.names[nodes]:
				flag = True
				expeditorNode = nodes
				break
		if not flag:
			node = self.g.addNode()
			expeditorNode = node
			self.names[node] = expeditor			
			self.link[person] = node
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
	
	def propertiesToTulipProperties(self):
		for person, node in self.link.items():
			self.receivedMails[node] = self.users[person]["received_mails"]
			self.sentMails[node] = self.users[person]["sent_mails"]
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

def compute_cliqueNumber(liste, node):
    cpt = 0
    for clique in liste :
        if node in clique:
            cpt = cpt+1
    return cpt
    
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
	

def main(graph): 
    for edge in graph.getEdges():
        graph.delEdge(edge)
    for node in graph.getNodes():
        graph.delNode(node)
    nodeName = graph.getStringProperty("nodeName")
    #enronpath = "/net/cremi/cbadiola/travail/bioInfo/enron_mail_20110402/maildirtest/"
    enronpath = "C:/Users/admin/Downloads/enron_mail_20110402/"
    myParser = mailParser(graph, nodeName, enronpath)
    myParser.parse()
    auth = graph.getDoubleProperty("auth")
    hub = graph.getDoubleProperty("hub")
    hits(graph, auth, hub)#TODO: A re-normaliser


