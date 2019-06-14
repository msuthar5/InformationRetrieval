#Python 3.0
import re
import os
import collections
import time
import math
import string
import random
#import other modules as needed

class kmeans:
	def __init__(self,path):
		self.path = path
		self.clusters = {}
		self.doc_vectors = {}
		self.doc_lengths = {}
		self.inverted_index = {}
		self.id_doc_map = {}
		self.total_docs = 0
		# load stop words
		self.stop_words = open("time/TIME.STP","r+").read().lower().strip(" ").split("\n")
		# remove empty words
		self.stop_words = [term for term in self.stop_words if term is not '']
		#self.docs_by_query = None
		start = time.time()
		self.buildIndex()
		self.compute_tf_idf_values()
		end = time.time()
		self.k=2
		print("\nTF-IDF Index built in  {}  seconds.".format(end-start))

	"""
		This function implements K-Means Clustering on the set of documents
		by doing the following:

		- Generate k : centroids by selecting k sets of 3 random documents and taking the average
						of them

		- Clustering each document around the centroids

		- REPEAT till change in RSS is < 1000:
			- recompute each centroid as the average of its clustered documents
			- re-cluster every document
	"""
	def clustering(self,kvalue):
		start=time.time()
		# initialize cluster data structure
		self.clusters = { cluster:{"centroid":{},"closest_doc":[1000000,0],"members":[],"RSS":0 } for cluster in range(kvalue) }
		random_seeds = {}
		rand = None
		seed = len(self.id_doc_map.keys())
		# generate k centroids by randomly selecting sets of 3 documents
		# and taking the average of them
		for i in range(kvalue):
			# select randoms
			randoms = random.sample(range(seed),3)
			# add the vectors
			random_seeds[i] = self.add_vectors(randoms)
			# normalize the vector
			random_seeds[i] = self.multiply_vector(random_seeds[i],1/3)
			# store it as the ith cluster's centroid
			self.clusters[i]["centroid"] = random_seeds[i]
		# cluster all the documents around these clusters
		self.cluster_vectors()
		print("Training Model...\n")
		end=time.time()
		sum_rss = self.compute_rss()
		print("\nClusters built in  {}  seconds.".format(end-start))
		print("Total RSS: {}\n".format(sum_rss))
		new_sum_rss = sum_rss
		ct = 0
		# while change in rss > threshold, re-cluster
		while (abs(new_sum_rss-sum_rss) > 1000) or (ct==0):
		#while (new_sum_rss!=sum_rss) or (ct==0):
			start=time.time()
			self.recompute_centroids()
			self.cluster_vectors()
			end=time.time()
			sum_rss = new_sum_rss
			new_sum_rss = self.compute_rss()
			print("\nClusters built in  {}  seconds.".format(end-start))
			print("Total RSS: {}".format(sum_rss))
			print("Change in RSS: {}\n".format(abs(sum_rss-new_sum_rss)))
			ct += 1

		print("Clusters Converged")
		print("Clusters Model Complete")
		return new_sum_rss


	"""
		This function computes the sum rss of all the clusters
	"""
	def compute_rss(self):
		sum_rss = 0
		print("RSS Values")
		for k,v in self.clusters.items():
			print("cluster id: {} RSS: {} closest document id: {}".format(k,v["RSS"], v["closest_doc"][1]))
			sum_rss += v["RSS"]
		return sum_rss

	"""
		This function recomputes the centroids by taking the average
		of each cluster's documents
	"""
	def recompute_centroids(self):
		for k,v in self.clusters.items():
			# sum the cluster members
			centroid = self.add_vectors(v["members"],1)
			if len(v["members"]) != 0:
				# normalize the centroid
				self.clusters[k]["centroid"] = self.multiply_vector(centroid,1/len(v["members"]))
			else:
				self.clusters[k]["centroid"] = centroid
			# reset the cluster data structure
			self.clusters[k]["members"] = []
			self.clusters[k]["RSS"] = 0
			self.clusters[k]["closest_doc"] = [1000000000,0]

	"""
		The function clusters every document around the centroids
	"""
	def cluster_vectors(self):
		# loop through all the documents
		for doc_id in self.doc_vectors.keys():
			# get the closest centroid via euclidean_distance
			ret = self.closest_cluster(doc_id)
			closest=ret[0]
			distance=ret[1]
			# store the document in its cluster
			self.clusters[closest]["members"].append(doc_id)
			# if the documents is closer than the current closest document
			# set the closest document as the current doc
			if distance < self.clusters[closest]["closest_doc"][0]:
				self.clusters[closest]["closest_doc"][0] = distance
				self.clusters[closest]["closest_doc"][1]=doc_id

	"""
		This function returns the closest cluster centroid and its distance
		from it from the passed document id
	"""
	def closest_cluster(self,doc_id):
		closest = 0
		distance = -1
		# iterate over the cluster centroids
		for i in range(len(self.clusters.keys())):
			# get the distance from the document and the centroid
			n_d=self.euclidean_distance(self.clusters[i]["centroid"],self.doc_vectors[doc_id])
			if n_d < distance or distance == -1:
				closest = i
				distance = n_d
		# store add the document's distance to the RSS for the cluster
		self.clusters[closest]["RSS"] += math.pow(distance,2)
		return closest, distance

	"""
		This functino computers the euclidean_distance
		between the passed vectors
	"""
	def euclidean_distance(self,v1,v2):
		val = 0
		s1 = set(v1.keys())
		s2 = set(v2.keys())
		for k in list(s1.union(s2)):
			t1=t2=0
			if k in v1:
				t1 = v1[k]
			if k in v2:
				t2 = v2[k]
			val += math.pow((t1-t2),2)
		return math.sqrt(val)

	"""
		This function iterates through the inverted_index and computes the
		tf-idf score for each term for every document and also creates the vectors for
		every document (required for Rocchio algorithm)

		we also compute the length of every document
	"""
	def compute_tf_idf_values(self):
		#all_docs = set(self.doc_vectors.keys())
		for term in self.inverted_index.keys():
			#updated = []
			p_list = self.inverted_index[term]
			# compute the idf
			idf = math.log10(self.total_docs/len(p_list[1:]))
			# save the idf
			self.inverted_index[term][0] = idf
			for idx in range(1,len(p_list)):
				# compute the tf
				tf = 1+math.log10(self.inverted_index[term][idx][1])
				# save the tf
				self.inverted_index[term][idx][1] = tf
				# save the tf-idf score for the term in the document's vector
				self.doc_vectors[p_list[idx][0]][term] = tf*idf
				# increment by doc_length for the current document
				#self.doc_lengths[ p_list[idx][0] ] += math.pow( tf*idf ,2)
				#updated.append(p_list[idx][0])
			"""to_update = list( all_docs.difference(set(updated)) )
			for id in to_update:
				self.doc_vectors[id][term] = 0"""
		# iterate through the doc_lengths structure and normalize it
		"""for doc_id, val in self.doc_lengths.items():
			self.doc_lengths[doc_id] = math.sqrt(val)"""

	"""
		This function builds the inverted_index by reading in a token
		at a time
	"""
	def buildIndex(self):
		content = open(self.path,"r+").readlines()
		current_doc = None
		pos_ctr = 0
		kennedy = 0
		# parse the document line by line
		for line in content:
			# splite the line into a list of words
			terms = line.split(" ")
			# if the sentinal *TEXT is found, this means we have a new document
			if terms[0] == "*TEXT":
				# update the current document variable to the id of the document
				current_doc = terms[1]
				# create an empty document vector for the new document
				self.doc_vectors[current_doc] = {}
				self.doc_lengths[current_doc] = 0
				self.id_doc_map[self.total_docs]=current_doc
				self.total_docs += 1
				# resent the position counter to 0
				pos_ctr = 0
			else:
				# parse each term
				for term in terms:
					# convert it into a valid token
					term = self.tokenize_word(term)
					if (term not in self.stop_words) and (len(term) > 0):
						# add the term to the inverted index
						self.add_to_index(term, pos_ctr, current_doc)
					pos_ctr += 1

	"""
		This function adds a term to the index.
			term    - the word to be added
			doc_id  - the id of the document
			pos_ctr - the position where the word was found in the document
	"""
	def add_to_index(self,term,pos_ctr,doc_id):
		"""
			We use a Try Catch block for efficiency.
			If the lookeup: self.inverted_index[term] fails, this means the term
			is not in the dictionary so we must create a new entry for it. If we did not
			use a try catch block, we would have to iterate over the entire index to see if the term is
			already accounted for
		"""
		try:
			# O(1) lookup that will either fail or is in the index
			p_list = self.inverted_index[term]
			found = 0
			for tup_idx in range(1,len(p_list)):
				# check if the doc_id is already in the posting list of the term
				if p_list[tup_idx][0] == doc_id:
					# increment the tf by 1
					self.inverted_index[term][tup_idx][1] += 1
					# save the position in the posting list
					self.inverted_index[term][tup_idx][2].append(pos_ctr)
					found = 1
			# if the posting list for the term does not contain the document
			if found == 0:
				# create new entry in posting list
				self.inverted_index[term].append( [doc_id,1,[pos_ctr]] )
		# create new entry in the inverted index for the term
		except KeyError as e:
			self.inverted_index[term] = [0,[doc_id,1,[pos_ctr]]]

	"""
		This function tokenizes the passed word and converts it into a term
		we use a try and catch to figure out of the ret_word is an integer string
		or not. If it is not, it is tokenized in the exception clause
	"""
	def tokenize_word(self,word):
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
		This function does scalar multiplication on a vector
		and return the vector
	"""
	def multiply_vector(self,vector,scalar):
		return {k:scalar*v for k,v in vector.items()}

	"""
		This function adds document_vectors
		documents: a list of document_ids to add
	"""
	def add_vectors(self,documents,flag=0):
		# initialize the result vector that will be returned
		res = {}
		for doc_id in documents:
			doc_1 = res
			# load the document vector
			if flag == 1:
				doc_2 = self.doc_vectors[doc_id]
			else:
				doc_2 = self.doc_vectors[self.id_doc_map[doc_id]]
			keys_1 = set(doc_1.keys())
			keys_2 = set(doc_2.keys())
			# compute the addition for all terms in (doc_1.terms UNION doc_2.terms)
			for term in list( keys_1.union(keys_2) ):
				t1 = 0
				t2 = 0
				if term in doc_1:
					t1=doc_1[term]
				if term in doc_2:
					t2=doc_2[term]
				res[term] = t1+t2
		return res

if __name__ == '__main__':
	a = kmeans("time/TIME.ALL")
	a.clustering(5)
