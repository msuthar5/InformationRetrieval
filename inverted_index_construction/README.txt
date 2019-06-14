this program is written assuming it will be run with python2.7

The program is run as a module and contains the 6 test queries that are located in: Output.txt

it can either be imported via: "from index import *"
- and an object can be instantiated via: "object = index("collection/")"
  -> it is assumed that collection/ is at the same directory level as the program

it can also be run directly through the terminal via: "python2.7 index.py"

MERGE ALGORITHM

the merge algorithm mimics the merge algorithm that the professor covered in lecture (slide 20 of lecture 2)

I generalized this algorithm to work on an arbritrary number of query terms as follows:

(1) I iterate over the positing lists of the query terms in the inverted index and extract the document_ids into a list and store
    it into the dictionary called "documents"

   documents is a dictionary of { term1: [doc_id1, doc_id2, ....], ...}

(2) use an O(n log n) sorting algorithm to sort the document ids list in increasing order

(3) I then compute the smallest sized entry from documents and iterate over this to find matches

(4) I generate a list of pointers of the form: [current_idx, term, last_idx + 1]

    where each entry represents where we are at in iterating over the terms list of doc_ids
    this is to generalize the merge algorithm over an variable number of terms

(5) Now that we have all our required data structures, we can compute the intersection as follows:

    - iterate over the list found in (3), call this list X
      - iterate over the doc_ids list of the other terms, call the other term Y
        - while the pointer to Y is < the pointer to X, increment the pointer
        - if the pointer to X == the pointer to Y, then the term is common in X and Y
          - increment the counter that is tracking how documents contain the term
          - if the number of documents that contain the term is equal to the number of documents
            in the documents data structure, then ADD THE DOC_ID TO THE RETURN LIST

So this algorithm does exactly as the algorithm we covered in class does, except that it
is generalized for an arbritrary number of terms.
