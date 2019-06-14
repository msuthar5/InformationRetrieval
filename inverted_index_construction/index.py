#Python 2.7.3
import re
import os
import collections
import time

class index:
	def __init__(self, path):
		self.inverted_index = {}
		self.term_id_table = {}
		self.path = path
		# list of VERY simple stop words to make sure are not in the dictionary
		#self.stop_words = ["and", "or", "a", "the", "an", "are", "is", "i"]
		# get the time to build the index
		self.stop_words = []
		start = time.time()
		self.buildIndex()
		end = time.time()
		print("\nIndex built in {} seconds.".format(end-start))
		print(self.inverted_index["yemen"])

	def buildIndex(self):
		# get a list of the files in the directory pointed to by self.path
		files = os.listdir(self.path)
		# local variables used in building the index
		document_file = None
		data = None
		posting_list = None
		term = None
		term_id = 0

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
								self.add_to_index(term, doc_id, pos_ctr, term_id)
						pos_ctr += 1
						term_id += 1

	"""
		This function adds a term to the index.
			term    - the word to be added
			doc_id  - the id of the document
			pos_ctr - the position where the word was found in the document
	"""
	def add_to_index(self, term, doc_id, pos_ctr, term_id):
		"""
			We use a Try Catch block for efficiency.
			If the lookeup: self.inverted_index[term] fails, this means the term
			is not in the dictionary so we must create a new entry for it. If we did not
			use a try catch block, we would have to iterate over the entire index to see if the term is
			already accounted for
		"""
		#self.term_id_table[term_id] = term
		try:
			# O(1) lookup that will either fail or is in the index
			self.inverted_index[term]
			# checker to see if the term was found already in the CURRENT document
			found = 0
			# Iterate over the posting list for the term to see if term is already in the CURRENT document
			for tup_idx in range(0,len(self.inverted_index[term])):
				if self.inverted_index[term][tup_idx][0] == doc_id:
					self.inverted_index[term][tup_idx][1].append(pos_ctr)
					found = 1
			# if term not already in the posting list for any existing documents
			# create a new posting list for the term
			if found == 0:
				self.inverted_index[term].append((doc_id,[pos_ctr]))
		except:
			# if the lookup self.inverted_index[term] resulted in a key error,
			# create a new entry in the inverted index for the term
			self.inverted_index[term] = [(doc_id,[pos_ctr])]


	# implement efficient pointer query to merge terms
	def and_query(self, query_terms):
		start = time.time()
		documents = {}
		posting_list = []
		ret_list = []
		# initialize the min len to infinity
		min_len = (999999999,None)
		# list of pointers for each posting list we are iterating over
		pointers = []
		ctr = 0
		# iterate over query_terms
		for term in query_terms:
			try:
				# iterate over each entry in the posting list
				for tup in self.inverted_index[term]:
					# add the document_id to the posting list
					posting_list.append(int(tup[0]))

				# O(n log(n)) sort both in worst and average case
				# we sort so we can efficiently perform pointer based searches
				posting_list.sort()
				# documents is a dictionary of { term: [doc_id1, doc_id2, ....]}
				documents[term] = posting_list
				# loop to find the smallest posting list
				if len(posting_list) < min_len[0]:
					min_len = (len(posting_list), term)
				# reinitialize the posting list for next iteration
				posting_list = []
			except:
				end = time.time()
				self.print_results([], query_terms, (end-start))
				return "ERROR TERM: {} NOT IN THE INDEX".format(term)

		"""
			pointers is a list holding the pointers to each position
			in each posting list. Format: [current_idx, term, last_idx + 1]

			the first element in pointers is the pointer data structure for the
			smallest sized posting list
		"""
		pointers.append([0,min_len[1],len(documents[min_len[1]])])
		# add pointer structures for the remaining terms in the query
		for k,v in documents.items():
			if k != min_len[1]:
				pointers.append([0,k,len(v)])

		# this loop performs the actual pointer based traversal for merging the posting list
		# the outer loop is iterating over the indexes in the posting list for the
		# smallest sized posting list
		for min_idx in range(0,pointers[0][2]):
			# this loop iterates over all the other terms to basically
			# do a set intersection
			for data in range(1,len(pointers)):
				# advance pointer for the list to be compared until we are >= the value
				# that the pointer to the smallest posting list is pointing towards
				try:
					while documents[pointers[data][1]][pointers[data][0]] < documents[pointers[0][1]][min_idx] and pointers[data][0] < pointers[data][2]:
						# increment the pointer
						pointers[data][0] += 1
				except:
					print("None")
					end = time.time()
					self.print_results([], query_terms, (end-start))
					return
				# if there is an intersection between the min_list[idx] and to_be_compared_list[idx]
				# then increment the counter
				if documents[pointers[0][1]][min_idx] == documents[pointers[data][1]][pointers[data][0]]:
					ctr += 1
				# if the term is present in every document, then it is a hit and
				# add it to the return list
				if ctr == len(pointers) - 1:
					ret_list.append(documents[pointers[0][1]][min_idx])
			# reinitialize ctr for the next term.
			ctr = 0
		end = time.time()

		self.print_results(ret_list, query_terms, (end-start))
		return ret_list

	"""
		This function prints the results of the executed query_terms
			- p_list      - the list of document ids
			- query_terms - the list of terms that were in the query
			- time        - time it took to execute the query
	"""
	def print_results(self, p_list, query_terms, time):
		print("")
		st = "Results for the Query: "
		# build the query string
		for a in range(0,len(query_terms)-1):
			st += " {} AND".format(query_terms[a])
		st += " {}".format(query_terms[len(query_terms)-1])
		print(st)
		print("Total Docs retrieved: {}".format(len(p_list)))
		# print the list of documents
		for doc_id in p_list:
			# format the file name
			print("Text-{}.txt".format(doc_id))
		print("Retrieved in: {} seconds".format(time))
		print("")

	def print_dict(self):
		# sort the keys in the inverted_index
		for k in sorted(self.inverted_index):
			# print the term and its posting list
			print("{} {}".format(k,self.inverted_index[k]))

	def print_doc_list(self):
		# get a list of files we did the index over
		files = os.listdir(self.path)
		for file in files:
			# format the string
			print("Doc ID: {} ==> {}".format(re.search(r'\d+', file).group(), file))

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

if __name__ == '__main__':
	a = index("collection/")
	# Same 6 test queries from Output.txt
	#x=a.and_query(['yemeni', 'yemen'])
	#x=a.and_query(['approached', 'terrorist'])
	#x=a.and_query(['said', 'terrorist', 'general'])
	#x=a.and_query(['assorted', 'appreciate', 'kenya'])
	#x=a.and_query(['spreading', 'general'])
	#x=a.and_query(['army', '1960', 'along'])
	x=a.and_query(['army', 'military', 'emerging', 'europeans'])
