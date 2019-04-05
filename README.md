# pyspark_utils

task specific utilities functions for pyspark use cases.

## How to use ?

* indivisual notebook contains a specific utilities whose description and use case are mentioned below


1. very fast connected component finder in spark/pyspark
  
   Notebook: very_fast_connected_component_finder.ipynb
    
      -> In many cases we have connection edges between the items in pyspark and we will need to find the connected components within those graphs. However with very very large volme of edges and vertices (billions) , which is present in a distributed file system such as RDD data, above program helps to find the connected components.
    
    -> INPUT: find_connected_components(sc, edges, n_dist = 5) , 
        where sc : SparkContext
        edges: RDD of format (n1, n2) where n1 and n2 are the two vertices of the edges.
        n_dist: distance upto which we want to find connected components.
        
        
    -> OUTPUT: returns rdd of format  (x, [c1, c2, c3, c4 ...ck])
        where x is the prime node and c1, c2, c3....ck are connected components.
        

