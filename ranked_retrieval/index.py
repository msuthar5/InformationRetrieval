#Python 3.0
import re
import os
import collections
import time
import math
import operator
import random
import matplotlib.pyplot as plt
import matplotlib as mpl
#import other modules as needed

class index:
	def __init__(self,path):
		self.inverted_index = {}
		# maps terms to term ids
		self.term_id_table = {}
		self.path = path
		self.tid_counter = 0
		self.total_docs = 0
		# store stop words as a list
		self.stop_words = open("stop-list.txt").read().split("\n")
		# storage for champions list
		self.champion_list = {}
		start = time.time()
		self.buildIndex()
		self.compute_idf_values()
		# compute and store the champions list for every term
		self.compute_champions_list()
		end = time.time()
		self.id_term_table = {v: k for k, v in self.term_id_table.items()}
		print("\nIndex built in {} seconds.".format(end-start))

	"""
		This function computes the term_frequency and inverted_document_frequency
		values for every term and document in the dictionary

		We kept a running count of the term frequency while constructing the index
	"""
	def compute_idf_values(self):
		for key,val in self.inverted_index.items():
			# compute idf value for each term; len(val)-1 represents the number of documents
			# containing the term
			val[0] = math.log10(self.total_docs/(len(val) - 1))
			# iterate through the posting list and compute weighted-term-frequency
			# for each term
			for item in range(1,len(val)):
				# store log term frequency
				val[item][1] = 1 + math.log10(val[item][1])

	"""
		the value for 'r', the # of documents in the list for a term, is variables
		and is dependent on the size of the posting list for the given term.

		It does not make sense to use the same r-value for every term, because
		rare terms might not even have r documents.
		So, the r value is the top 30%  of the documents in that term's posting list
	"""
	def compute_champions_list(self):
		for term, posting_list in self.inverted_index.items():
			# temporary storage of documents in the posting list
			top_docs = {}
			# iterate over posting list and store the weighted-term-frequency
			# for each document
			for i in range(1, len(posting_list)):
				top_docs[posting_list[i][0]] = posting_list[i][1]
			# sort the top_docs dictionary in reverse order
			sorted_top_docs = sorted(top_docs, key=top_docs.get, reverse=True)
			# select 0.3 of the documents with highest weighted-term-frequency
			self.champion_list[term] = sorted_top_docs[:math.ceil((len(posting_list)-1) * 0.30)]

	"""
		This function uses the index elimination method to
		return similar documents to the query
	"""
	def inexact_query_index_elimination(self, query_terms, k):
		start = time.time()
		# build the query vector
		query_vector = self.query_to_vector(query_terms)
		# term variables for storage
		document_vectors = {}
		empty_vector = {}
		q_ = {}

		# sort the query_vector by decreasing tf-idf values
		sorted_qv = sorted(query_vector, key=query_vector.get, reverse=True)
		# take the top half of the query terms
		top_half = sorted_qv[:math.ceil(len(sorted_qv)/2)]

		# copy the top half terms and their tf-idf values into the new query_vector: q_
		for term in top_half:
			q_[term] = query_vector[term]

		# Build the document vectors for each document
		for i in range(1,self.total_docs + 1):
			doc_vec = {}
			for key in q_.keys():
				doc_vec[key] = 0
			# store the vector with its associated document_id
			document_vectors[str(i)] = doc_vec
		# compute the tf-idf value for each term in the vectors
		document_vectors = self.build_document_vectors(q_,document_vectors)

		# build an empty vector to prevent doing cosine_similarity
		# on a vector of all 0's
		for key in q_.keys():
			empty_vector[key] = 0
		for key, val in document_vectors.items():
			if val == empty_vector:
				document_vectors[key] = 0
			else:
				# compute and store cosine similarity for the document
				document_vectors[key] = self.cosine_similarity(q_, val)
		# python std library sorting function for dictionary
		# we specify that we went to sort by value
		# and sort in reverse order because higher cosine similarities are associated with smaller angles
		sorted_document_vectors = sorted(document_vectors, key=document_vectors.get, reverse=True)
		end = time.time()

		self.print_query_results(query_terms,sorted_document_vectors[:k], document_vectors,end-start, "Index Elimination")
		return (end-start)

	"""
		This method implements cluster pruning

		1- We compute sqrt(# docs) RANDOM leaders
		2- We cluster every other document to their nearest leader from (1)
		3- We compute the closest leader to the query vector via cosine similarity
		4- We compute the k closest documents to the query from the
			documents associated with closest leader from (3)
	"""
	def inexact_query_cluster_pruning(self, query_terms, res_size):
		start = time.time()
		# compute the number of leaders as sqrt(total_docs)
		num_leaders = math.ceil(math.sqrt(self.total_docs))
		# get a random list of num_leaders leaders
		leaders = random.sample(range(1,self.total_docs), num_leaders)
		document_vectors = {}
		clusters = {}
		empty_vector = {}

		# generate query vector
		query_vector = self.query_to_vector(query_terms)

		# generate the empty vector
		for key in query_vector.keys():
			empty_vector[key] = 0
		# generate the document vectors for every document
		for i in range(1,self.total_docs + 1):
			doc_vec = {}
			for key in query_vector.keys():
				doc_vec[key] = 0
			document_vectors[str(i)] = doc_vec
		# compute tf-idf value for every query term in the documents
		document_vectors = self.build_document_vectors(query_vector,document_vectors)

		# build data structure to hold followers for each leader
		# format: {leader_doc_id: [ leader_doc_vector, { dictionary of followers} ],... }
		for leader in leaders:
			clusters[str(leader)] = [document_vectors[str(leader)], {}]

		# This loop places every document into their associated clusters
		# Using cosine similarity
		for key,val in document_vectors.items():
			scores = {}
			# only cluster documents that are NOT already leaders
			if key not in leaders:
				for k,v in clusters.items():
					# immedietly assign cosine score of 0 if either the
					# leader of the document vector is the empty vector
					if val == empty_vector or v[0] == empty_vector:
						scores[k] = 0
					else:
						# compute and store cosine similarity
						scores[k] = self.cosine_similarity(val, v[0])
				# sort scores
				sorted_scores = sorted(scores, key=scores.get, reverse=True)
				# cluster the document into its leaders cluster
				clusters[str(sorted_scores[0])][1][key] = val

		query_scores = {}
		# This loop finds the closest leader by using cosine_similarity
		# against the query_vector and cluster_leader_vector
		for key,val in clusters.items():
			if val[0] == empty_vector:
				query_scores[key] = 0
			else:
				query_scores[key] = self.cosine_similarity(query_vector, val[0])
		# sort the scores
		sorted_scores = sorted(query_scores, key=query_scores.get, reverse=True)
		# select the closest leader
		closest_leader = str(sorted_scores[0])
		next_closest = sorted_scores[1]

		# store all the documents in the leader's cluster
		cluster_dvs = clusters[closest_leader][1]

		# This loop will compute the cosine_similarity between the query vector
		# and every vector in the leader's followers
		for k,v in cluster_dvs.items():
			if v == empty_vector:
				cluster_dvs[k] = 0
			else:
				# compute and store cosine similarity score
				cluster_dvs[k] = self.cosine_similarity(query_vector, v)

		# Sort the scores
		sorted_results = sorted(cluster_dvs, key=cluster_dvs.get, reverse=True)
		end = time.time()

		# print the top k results
		self.print_query_results(query_terms,sorted_results[:res_size], cluster_dvs,end-start, "Cluster Pruning")
		return (end - start)

	"""
		This method implements the champions list method for inexact retrieval
	"""
	def inexact_query_champion(self, query_terms, k):
		start = time.time()
		query_vector = {}
		document_vectors = {}
		champ_list_vectors = {}

		# we do a set union of all the documents in the champions list
		# of every query term
		s = set()
		# iterate over the champions list for each term
		for term in list(set(query_terms.split(" "))):
			if term in self.term_id_table:
				# get the champions list for the term
				champ_list = self.champion_list[self.term_id_table[term]]
				# set union
				s = s.union(set(champ_list))
		# build query vector
		query_vector = self.query_to_vector(query_terms)
		# build the document vectors
		for i in range(1,self.total_docs + 1):
			doc_vec = {}
			for key in query_vector.keys():
				doc_vec[key] = 0
			document_vectors[str(i)] = doc_vec
		# compute tf-idf value for the document vectors
		document_vectors = self.build_document_vectors(query_vector,document_vectors)

		# extract the documents in the champions lists from the total document vectors
		for doc_id in list(s):
			champ_list_vectors[doc_id] = document_vectors[doc_id]

		# compute cosine similarity between query vector and each document vector
		# in the champions list
		for key, val in champ_list_vectors.items():
			champ_list_vectors[key] = self.cosine_similarity(query_vector, val)
		# sort the scores
		sorted_champ_vectors = sorted(champ_list_vectors, key=champ_list_vectors.get, reverse=True)
		end = time.time()
		# print the top k documents
		self.print_query_results(query_terms,sorted_champ_vectors[:k], champ_list_vectors,end-start, "Champion List")
		return (end - start)

	def buildIndex(self):
		#function to read documents from collection, tokenize and build the index with tokens
		# implement additional functionality to support methods 1 - 4
		#use unique document integer IDs
		# get a list of the files in the directory pointed to by self.path
		files = os.listdir(self.path)
		self.total_docs = len(files)
		# local variables used in building the index
		document_file = None
		data = None
		posting_list = None
		term = None

		# iterate over every file
		for file in files:
			# regex to get the id out of the filename
			# example: the id for Text-409.txt is 409
			doc_id = re.search(r'\d+', file).group()
			# open the file
			document_file = open( self.path + file,"r+")
			# convert every string of lines into a list of lines
			lines = document_file.readlines()
			document_file.close()
			# pos_ctr is the counter that holds the position of the word in the document
			pos_ctr = 1

			# iterate over every line in the file
			for line in lines:
				# convert every line into a list of of words separated by a SINGLE space
				for word in line.split(" "):
					# tokenize the word and convert it into a term to be added into the index
					term = self.tokenize_word(word)
					# if the term is valid after tokenization, then the len(term) > 0
					# add the term to the dictionary
					if term != None:
						# make sure the term is not a stop word
						if term not in self.stop_words:
							if len(term) > 0:
								# add the valid term to the inverted index
								self.add_to_index(term, doc_id, pos_ctr)
						pos_ctr += 1

	"""
		This function tokenizes the passed word and converts it into a term
		we use a try and catch to figure out of the ret_word is an integer string
		or not. If it is not, it is tokenized in the exception clause
	"""
	def tokenize_word(self, word):
		# remove all non alphanumeric literals
		word = re.sub(r'[^\w]', '', word)
		try:
			# if it is not an integer string, this line will fail and we will
			# jump to except clause
			ret_word = int(word)
			ret_word = str(ret_word)
		except:
			# tokenize the non-int string
			ret_word = re.sub(r'[\d]','', word).lower()
		return ret_word

	"""
		This function adds a term to the index.
			term    - the word to be added
			doc_id  - the id of the document
			pos_ctr - the position where the word was found in the document
	"""
	def add_to_index(self, term, doc_id, pos_ctr):
		"""
			We use a Try Catch block for efficiency.
			If the lookeup: self.inverted_index[term] fails, this means the term
			is not in the dictionary so we must create a new entry for it. If we did not
			use a try catch block, we would have to iterate over the entire index to see if the term is
			already accounted for
		"""
		try:
		#if term in self.inverted_index:

			# O(1) lookup that will either fail or is in the index
			self.term_id_table[term]
			tid = self.term_id_table[term]
			found = 0
			# checker to see if the term was found already in the CURRENT document

			#posting_list = self.inverted_index[tid]
			for tup_idx in range(1,len(self.inverted_index[tid])):
				if self.inverted_index[tid][tup_idx][0] == doc_id:
					self.inverted_index[tid][tup_idx][1] += 1
					self.inverted_index[tid][tup_idx][2].append(pos_ctr)
					found = 1

			if found == 0:
				self.inverted_index[tid].append([doc_id,1,[pos_ctr]])
		# term is not in the dictionary so add it
		except KeyError as e:
		#else:
			self.term_id_table[term] = len(self.term_id_table.keys()) + 1
			self.inverted_index[self.term_id_table[term]] = [0,[doc_id,1,[pos_ctr]]]

	def query_to_vector(self,query):
		# convert set to identify unique entries
		q = list(set(query.split(" ")))
		q_ = {}
		query_vector = {}
		# compute and store document frequency for query vector
		for term in q:
			ctr = 0
			for val in query.split(" "):
				if term == val:
					ctr += 1
			# store the counter for the term
			q_[term] = ctr
			ctr = 0
		# create the vector for the query
		# vector has form {term1: tf-idf, term2: tf-idf,...}
		# the vector is a dictionary so allow for lookups when computing cosine similarity
		# between query_vertor and document_vector
		for key,val in q_.items():
			if key in self.term_id_table:
				q_[key] = ((1 + math.log10(val)) * self.inverted_index[self.term_id_table[key]][0])
			else:
				q_[key] = 0
		"""for key,val in q_.items():
			q_[key] = ((1 + math.log10(val)) * self.inverted_index[self.term_id_table[key]][0])"""
		#print(q_)
		print(q_)
		return(q_)

	"""
		This function builds the document vectors by computing the tf-idf
		value for every term in the vectors in document_vectors
	"""
	"""def build_document_vectors(self, query_vector, document_vectors):
		for term in query_vector.keys():
			# load the posting list
			if term in self.term_id_table:
				posting_list = self.inverted_index[self.term_id_table[term]]
				# we already computed the weighted-term-frequency and inverted-document-frequency
				# while constucting the index, so all we do is multiply them together
				for i in range(1, len(posting_list)):
					# calculate and store tf-idf value
					document_vectors[posting_list[i][0]][term] = posting_list[i][1] * posting_list[0]
			else:
				for doc_id in range(1,self.total_docs+1):
					document_vectors[str(doc_id)][term] = 0
		return document_vectors"""

	"""def build_document_vectors(self, query_vector, document_vectors):
		# build the document vectors for all the query terms
		for key in query_vector.keys():
			if key in self.term_id_table:
				for entry in self.inverted_index[self.term_id_table[key]][1:]:
					document_vectors[entry[0]][key] = self.inverted_index[self.term_id_table[key]][0] * entry[1]

		# build the document vectors for all terms not in the query terms
		for k,v in self.inverted_index.items():
			idf = v[0]
			posting_list = v[1:]
			term = self.id_term_table[k]
			if term not in query_vector.keys():
				for idx in posting_list:
					document_vectors[ idx[0] ][ term ] = idf * idx[1]
		return(document_vectors)"""

	def build_document_vectors(self, query_vector, document_vectors):
		for k,v in self.inverted_index.items():
			idf = v[0]
			term = self.id_term_table[k]
			posting_list = v[1:]
			for idx in posting_list:
				document_vectors[ idx[0] ][term] = (idx[1] * idf)

		for key in query_vector.keys():
			if key in self.term_id_table:
				for entry in self.inverted_index[self.term_id_table[key]][1:]:
					document_vectors[entry[0]][key] = self.inverted_index[self.term_id_table[key]][0] * entry[1]

		return document_vectors


	"""
		This method implements the exact query retrieval method

		We generate a vector out of every document and select the top K
		with the highest cosine_similarity when compared to the query vector
	"""
	def make_query_vector_equal_length(self,query_vector,vector_2):
		for key in vector_2.keys():
			if key not in query_vector:
				query_vector[key] = 0

		for key in query_vector.keys():
			if key not in vector_2:
				vector_2[key] = 0
		return query_vector, vector_2

	def exact_query(self, query_terms, k):
		start = time.time()
		# build the query vector
		query_vector = self.query_to_vector(query_terms)
		document_vectors = {}
		empty_vector = {}
		# build the document vectors
		for i in range(1,self.total_docs + 1):
			doc_vec = {}
			for key in query_vector.keys():
				doc_vec[key] = 0
			document_vectors[str(i)] = doc_vec
		# compute tf-idf value for the document vectors
		document_vectors = self.build_document_vectors(query_vector,document_vectors)
		print("DV",document_vectors['10'])

		# build the empty vector
		for key in query_vector.keys():
			empty_vector[key] = 0
		for key, val in document_vectors.items():
			# assign score of 0 if vector is the empty vector
			if val == empty_vector:
				document_vectors[key] = 0
			else:
				q_,val = self.make_query_vector_equal_length(query_vector, val)
				document_vectors[key] = self.cosine_similarity(q_, val)
		# python std library sorting function for dictionary
		# we specify that we went to sort by value
		# and sort in reverse order because higher cosine similarities are associated with smaller angles
		sorted_document_vectors = sorted(document_vectors, key=document_vectors.get, reverse=True)

		end = time.time()
		self.print_query_results(query_terms,sorted_document_vectors[:k], document_vectors,end-start, "Exact Retrieval")
		return (end - start)

	"""
		This method simply prints the results of the query and the method used
	"""
	def print_query_results(self, query_terms, documents, dvs, time, method_used):

		print("\nresults for top {} results for the query: '{}':".format(len(documents),query_terms))
		print("Retrieval Method: {}".format(method_used))
		for i in range(len(documents)):
			print("\t({}): {} --> cosine_similarity score: {}".format(i+1, "Text-{}.txt".format(documents[i]) , dvs[documents[i]]))

		print("time to compute: {}".format(time))


	"""
		This function computes the cosine similarity score for the query vector
		and the document vector
	"""
	def cosine_similarity(self, query_vector, document_vector):
		# init variables
		qv_length = 0
		dv_length = 0
		score = 0

		# compute the length of each vector
		for key,val in query_vector.items():
			qv_length += math.pow(val,2)
			dv_length += math.pow(document_vector[key],2)

		# taking the square root of the vector
		qv_length = math.sqrt(qv_length)
		dv_length = math.sqrt(dv_length)

		# we use the key to compute the dot product because the dictionary
		# is not ordered, so we use keys to ensure we compute correctly
		for key,val in query_vector.items():
			score += (query_vector[key] * document_vector[key])

		score = score / (qv_length * dv_length)
		return(score)

	"""
		This function runs a set of test queries and graphs the data accordingly

		this function uses matplotlib and seaborn for visualizations
	"""
	def performance_measure(self):

		# list of queries to run
		queries = ["yemeni yemen",
					"assorted kenya africa",
					"spreading general",
					"army 1960 along",
					"army military emerging europeans",
					"said terrorist general"]

		# build the data structure to hold all the times taken to run each query
		# for each method of retrieval
		times = {}
		times["exact"] = []
		times["inexact index elimination"] = []
		times["inexact champtions list"] = []
		times["inexact clustering"] = []

		# Run each query in each of the 4 retrieval methods and store
		# the time taken to compute the results
		for i in range(len(queries)):
			times["exact"].append( self.exact_query(queries[i],5))
			times["inexact index elimination"].append(self.inexact_query_index_elimination(queries[i],5))
			times["inexact champtions list"].append(self.inexact_query_champion(queries[i],5))
			times["inexact clustering"].append(self.inexact_query_cluster_pruning(queries[i],5))

		# the x-coordinates corresponding to the query number
		a = [1,2,3,4,5,6]
		mpl.style.use("seaborn")
		plt.title("Retrieval Method Performance Evaluation", fontsize=16, fontweight='bold')
		plt.xlabel("Query number")
		plt.ylabel("Time to execute query (seconds)")
		i = 0
		for k,v in times.items():
			# plot the timings for each method of retrieval and color the line accordingly
		    plt.plot(a,v, 'C'+str(i+2), label=k)
		    i += 1
		plt.legend(loc='upper left')
		# print the plot
		plt.show()


if __name__ == '__main__':
	a = index("collection/")
	#a.exact_query("BACKGROUND OF THE NEW CHANCELLOR OF WEST GERMANY LUDWIG ERHARD".lower(), 10)
	a.exact_query("yemen without", 10)
	#a.inexact_query_champion("assorted kenya africa",5)
	#a.inexact_query_cluster_pruning("assorted kenya africa",5)
	#a.performance_measure()
	#print(len(a.inverted_index[a.term_id_table["yemen"]])-1)
	#print(a.inverted_index[a.term_id_table["yemen"]])
	#print(len(a.inverted_index[a.term_id_table["without"]])-1)
	#print(a.inverted_index[a.term_id_table["without"]])
