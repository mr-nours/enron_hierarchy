from tulip import *
from email.parser import Parser
from email.utils import parsedate_tz
from email.utils import mktime_tz
from datetime import datetime
from datetime import timedelta
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
   		# For each mail received
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

def main(graph): 
	for edge in graph.getEdges():
		graph.delEdge(edge)
	for node in graph.getNodes():
		graph.delNode(node)
	nodeName =  graph.getStringProperty("nodeName")
	myParser = mailParser(graph, nodeName, "/net/cremi/cbadiola/travail/bioInfo/enron_mail_20110402/maildirtest/")
	myParser.parse()
