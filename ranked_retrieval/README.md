Comparison of Inexact retrieval methods with inexact retrieval

To compare the performance of the exact retrieval vs inexact retrieval methods, I created a linear chart that runs a preselected set of queries on each of the retrieval methods and plots the time taken for each query.

Queries:

Query Number	Query
1     "yemeni yemen"
2     "assorted kenya africa"
3     "spreading general"
4     "army 1960 along"
5     "army military emerging europeans"
6     "said terrorist general"

Resulting Graph:
 
This graph was created using Matplotlib and Seaborn.




Evaluation:

As expected, the inexact cluster pruning retrieval method took the most time on each query. 
Order of the methods by decreasing (average) time taken for each query:
1.    Inexact Cluster Pruning
2.    Exact
3.    Inexact Index Elimination
4.    Inexact Champion’s List

These results are very consistent with the underlying theory behind each algorithm. The inexact cluster pruning retrieval method took, by far, the most time to compute the top k results because this requires many cosine similarity operations to build the clusters. This makes sense, because we first have to find sqrt(423) random leaders, then cluster every single other document around their leader, then compute the leader for the query, and finally select the k closest documents from that given leader’s cluster. The next most expensive method was the exact retrieval method. This makes sense intuitively, because this method requires computing the 2nd most cosine similarity computations. Essentially, we compute a cosine similarity between the query vector and every single document vector. The last two methods are the index elimination and champions list method. This makes sense because index elimination does a lot more cosine similarities than the champion’s list does. The index elimination method is essentially the same as the exact retrieval method except that we trim down the size of the query vector by only keeping the 50% of terms with the highest IDF values. Champion’s List, as expected, takes the least amount of time as we perform the fewest cosine similarities on a predetermined list of documents for each term in our vocabulary.

The main take away is that the performance on each method is directly proportional to the number of cosine similarities we perform. It was extremely satisfying to see the results I expected as they are consistent with my understanding of the algorithm. This is the only reason why the clustering method is so much more expensive than any other method.

Implementation Details impacting performance:

The only area for bias was that I pre-selected 6 queries to run the tests on. Theoretically, it is possible that these queries are “outlier” cases that can potentially bias the results, but this is extremely unlikely. However, it seems appropriate to mention this. I use the same cosine similarity function for each similarity comparison, so there is no timing bias due to the cosine similarity algorithm I wrote.

Every query is represented as a string. I have code that converts the free text query into a vector.
So we call functions like: object.exact_query(query=”some free text string”, k=some_int)
 
•	Champion’s List Implementation
o	How I selected r
	I researched and found that it does not make sense to use a consistent r value for every term because of how rare terms might not even appear in  r documents. Instead, I used a consistent percentage of 30%. So the champions list for each term is the top 30% of terms by weighted log frequency
o	Implementation steps:
	We compute the champion’s list during index construction
	For each query term, we take do a set union of the champion’s list for each term
	Compute cosine similarity for each document and return top k documents by decreasing cosine similarity values
•	Index elimination
o	Implementation Steps
	For each query term, we get the inverted-document frequency and put these values in a list
	We then create a query vector from the top half of the query terms (terms with highest idf value)
	We then do cosine similarity between the new query vector and every document and return the top k of these documents
•	Cluster Pruning
o	Implementation Steps
	We randomly select sqrt(423) leaders and create vectors out of these documents
	For every other document (non-leaders), we cluster them with their closest leader 
•	Run cosine similarity on document and every leader
•	Cluster the document in the cluster of its closest leader
	Find the closest leader for the query vector
•	Run cosine similarity on the query vector and every leader vector
	Select the top k documents in the closest leader’s cluster based on cosine similarity between the query vector

