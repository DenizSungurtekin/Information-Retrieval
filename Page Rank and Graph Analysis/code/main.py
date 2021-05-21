import numpy as np
import networkx as nx

# rank of site u = sum(R(v)/L(v)) where each element are the liked page to u, L(v) number of link coming from v

# start with R(u), u a node of the system -> Start R(u) = 1/nbNode_in_system , compute ranking until convergence
# Represent the graph with matrix adj A(u,v) -> NxN, N = nbNode
# If there is a link u to v -> A(u,v) = 1 / L(u) else : A(u,v) = 0 -> not necesarily symetric
# Pange rank: R_k = A'R_k-1
# Convergece detected with l1 norm of R_k - R_k-1 -> sufficently close to zero -> stop


#Node class
class Node:
    def __init__(self,name,id,N):
        self.name = name
        self.hyperlinks = [] #List of node where self give the link
        self.id = id
        self.rank = 1/N
        self.givelink = [] #List of node which are giving the link of self
        self.L = 0

    def refresh(self):
        self.L = len(self.hyperlinks)


#Compute adjacency matrix
def computeAdj(nodes):
    N = len(nodes)
    A = np.zeros((N,N))
    for node in nodes:
        for othernode in nodes:
            if node == othernode:
                A[node.id][othernode.id] = 0
            elif othernode in node.hyperlinks:
                A[node.id][othernode.id] = 1/node.L
            else:
                A[node.id][othernode.id] = 0
    return A

#Compute page rank
def pageRank(A):
    R = [1/len(A) for i in range(len(A))]
    R = np.asarray(R)
    res = 10

    while res > 0.0000001:
        R_new = np.dot(np.transpose(A),R)
        res = np.linalg.norm((R_new-R),ord = 1)
        R = R_new

    return R

#Generate a random walk and compute the more visited nodes to define the rank
def generateGraphRandomTraversal(size,degree,walkLenght):
    G = nx.generators.random_graphs.barabasi_albert_graph(size,degree)
    pRank = nx.pagerank(G)
    random_node = np.random.choice(G.nodes)
    current = random_node
    visited = [current]
    for i in range(walkLenght):
        neighboor = []
        for element in G.edges(current):
            neighboor.append(element[1])
        next = np.random.choice(neighboor)
        visited.append(next)
        current = next

    visited = np.asarray(visited)
    visitedUnique = np.unique(visited)
    visited = visited.tolist()
    occurences = [("node: "+str(element)+" traversal Score : ",visited.count(element)) for element in visitedUnique]
    occurences.sort(key=lambda x:x[1])
    pRank = dict(sorted(pRank.items(), key=lambda item: item[1],reverse=True))

    return occurences[::-1],pRank




# Initialize graph
names = ["A","B","C","D","E","F","G"]
nodes = [Node(names[i],i,len(names)) for i in range(len(names))]
nodes[0].hyperlinks = [nodes[1],nodes[2],nodes[3]]
nodes[1].hyperlinks = [nodes[2]]
nodes[2].hyperlinks = [nodes[4]]
nodes[3].hyperlinks = [nodes[5],nodes[6]]
nodes[4].hyperlinks = [nodes[0]]
nodes[5].hyperlinks = [nodes[4]]
nodes[6].hyperlinks = [nodes[0]]
#
nodes[0].givelink = [nodes[4],nodes[6]]
nodes[1].givelink = [nodes[0]]
nodes[2].givelink = [nodes[0],nodes[1]]
nodes[3].givelink = [nodes[0]]
nodes[4].givelink = [nodes[2],nodes[5]]
nodes[5].givelink = [nodes[3]]
nodes[6].givelink = [nodes[3]]

#Refresh
for node in nodes:
    node.refresh()

A = computeAdj(nodes)

print("Adjacency matrix: ")
print(A)
R = pageRank(A)
for node in nodes:
    print(" ")
    print(node.name,"has a rank of",R[node.id],"with",len(node.givelink),"incomming links")

_,eigenVector = np.linalg.eig(np.transpose(A))
eigenVector = np.real(eigenVector)

magn = 0
for vect in eigenVector:
    if np.linalg.norm(vect) > magn:
        principalVect = vect
        magn = np.linalg.norm(vect)


#Explanation eigen - rank:

# since the pagerank algorithm is an iterative application of the link matrix,
# the ultimate pagerank vector will look a lot like an eigenvector associated with the highest eigenvalue of the link matrix.

# So back there we repeatedly applied the link matrix to the pagerank vector. This is equivalent
# to repeatedly multiplying the link matrix by itself, and then after enough self-multiplications, applying that result to the pagerank vector.
# But repeated multiplication... is exponentiation. So we can do that by exponentiating
# the matrix, which we can do by exponentiating the diagonal matrix with the eigenvalues in it.
# but this is a really, really big exponent. Hopefully an infinite exponent, in fact.
# So the diagonal matrix comes to be dominated by the largest (original) eigenvalue, and so too does the ultimate link-following matrix come to be dominated
# by a vector associated with the largest eigenvalue, and so the ultimate pagerank vector will be dominated that vector too.

print(" ")
print("Principal eigen Vector of A transpose:", principalVect)
print(" ")
print("Rank Vector:", R)

## 3.2
print("--------------Exercice 3.2----------------")
sizes = [100,500,1000]
degrees = [5,10,20]
walk = 10000 # The more the graph is big the more we need a big random walk size
for i in range(len(sizes)):
    traversal,pRank = generateGraphRandomTraversal(sizes[i],degrees[i],walk)
    print("-----------NEXT ITERATION----------------- ")
    print("Results for size =",sizes[i])
    print("Sorted rank by pageRank:", pRank)
    print("Sorted rank by random traversal:",traversal)


# traversals = the order in which the nodes are visited