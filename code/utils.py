import abcd_graph
from CAS import CASPostProcesser
from egosplit_sknetwork import EgoSplit
from Fstar import edge_clustering_to_node_clustering
from fast_hdbscan.hdbscan import clusters_from_spanning_tree
from numba import njit
import numpy as np
from partition_sknetwork import ECG
import scipy.sparse as sp
import sknetwork as sn

from tqdm import trange


import os

def run_abcd(
    n,
    xi,
    eta=1.,
    nout=0,
    gamma=2.5,
    delta=3,
    Delta=None,
    s=25,
    S=None,
    beta=1.5,
    d=8,
    rho=0.0,
    seed=42,
    julia="julia",
    folder=None,
    abcd_path=None,
):
    if Delta is None:
        Delta =  int(np.sqrt(n))
    if S is None:
        S = int(n**0.75)
    if eta == 1:
        params = abcd_graph.ABCDParams(
            vcount=n,
            xi=xi,
            num_outliers=nout,
            min_degree=delta,
            max_degree=Delta,
            gamma=gamma,
            min_community_size=s,
            max_community_size=S,
            beta=beta,
        )
        graph = abcd_graph.ABCDGraph(params).build()
        labels = np.empty(n, dtype="int32")
        for i, com in enumerate(graph.communities):
            labels[com.vertices] = i
        if nout > 0:
            labels[labels == np.max(labels)] = -1
        adjacency = graph.exporter.to_sparse_adjacency_matrix()
        return adjacency, labels

    # ABCDoo needs julia version
    if julia is None or folder is None or abcd_path is None:
        raise ValueError("Must pass julia, folder, and abcd_path if eta>1")
    config = f'''seed = "{seed}"                   # RNG seed, use "" for no seeding
n = "{n}"                   # number of vertices in graph
nout = "{nout}"                  # number of vertices in graph that are outliers
t1 = "{gamma}"                      # power-law exponent for degree distribution
d_min = "{delta}"                   # minimum degree
d_max = "{Delta}"                  # maximum degree
d_max_iter = "1000"           # maximum number of iterations for sampling degrees
t2 = "{beta}"                      # power-law exponent for cluster size distribution
c_min = "{s}"                  # minimum cluster size after growing
c_max = "{S}"                # maximum cluster size after growing
c_max_iter = "1000"           # maximum number of iterations for sampling cluster sizes
xi = "{xi}"                    # fraction of edges to fall in background graph
eta = "{eta}"                   # average number of community per community node
d = "{d}"                       # dimensionality of latent space
rho = "{rho}"                   # correlation between degree and number of communities node is in
degreefile = "deg_{seed}.dat"        # name of file do generate that contains vertex degrees
communitysizesfile = "cs_{seed}.dat" # name of file do generate that contains community sizes
communityfile = "com_{seed}.dat"     # name of file do generate that contains assignments of vertices to communities
networkfile = "edge_{seed}.dat" '''
    config_file = f'config_{seed}.dat'
    with open(config_file, 'w') as f:
        print(config, file=f)

    cmd = julia+" "+abcd_path+'abcd_oo_sampler.jl ' + folder+"/"+config_file
    os.system(cmd+' >/dev/null 2>&1')

    adjacency = sn.data.from_csv(f'edge_{seed}.dat')
    adjacency = adjacency[1:][:, 1:]

    list_of_coms = []
    n_coms = 0
    with open(f'com_{seed}.dat', "r") as file:
        for line in file:
            # each line has the format: node_id\t[a,b,c]\n
            # for this node belonging to communities a,b,c
            # if there are outliers then they have community [-1]
            # node_id counts up starting at 0
            x = line.split('\t')[1].rstrip()[1:-1] # gets a,b,c
            x_coms = [int(y) for y in x.split(',')]
            list_of_coms.append(np.array(x_coms))
            n_coms = max(n_coms, max(x_coms))

    labels = sp.lil_matrix((adjacency.shape[0], n_coms), dtype="bool")
    for x, x_coms in enumerate(list_of_coms):
        if x_coms[0] == 1:
            continue
        labels[x, x_coms - 2] = True
    labels = labels.transpose().tocsr()
    
    cmd = f'rm *_{seed}.dat'
    os.system(cmd+' >/dev/null 2>&1')

    return adjacency, labels


#############
# LC + HSLC #
#############

@njit(nogil=True)
def _get_data_index(indptr, indices, i, j):
    for index in range(indptr[i], indptr[i + 1]):
        if indices[index] == j:
            return np.int64(index)
    return np.int64(-1)


#@njit(nogil=True, fastmath=True)
def _lc_edge_similarity(
    adj_indptr,
    adj_indices,
    triu_indptr,
    triu_indices,
):
    lg_n_edges = 0
    for i in trange(len(adj_indptr)-1):
        neighborhood_size = adj_indptr[i+1] - adj_indptr[i]
        lg_n_edges += (neighborhood_size * (neighborhood_size-1)) // 2
    print(lg_n_edges)
    row = np.empty(lg_n_edges, dtype="int64")
    col = np.empty(lg_n_edges, dtype="int64")
    data = np.empty(lg_n_edges, dtype="float64")
    next_id = 0

    for i in trange(len(adj_indptr)-1):
        neighbors = adj_indices[adj_indptr[i]:adj_indptr[i+1]]
        for j_id in range(len(neighbors)):
            j = neighbors[j_id]
            j_neighborhood = set(adj_indices[adj_indptr[j]:adj_indptr[j+1]])
            e1_id = _get_data_index(triu_indptr, triu_indices, min(i,j), max(i,j))
            if e1_id == -1:
                print(i, j)
            for k_id in range(j_id):
                k = neighbors[k_id]
                k_neighborhood = set(adj_indices[adj_indptr[k]:adj_indptr[k+1]])
                e2_id = _get_data_index(triu_indptr, triu_indices, min(i,k), max(i,k))
                if e2_id == -1:
                    print(i, k)
                intersect = len(j_neighborhood.intersection(k_neighborhood))
                row[next_id] = e1_id
                col[next_id] = e2_id
                data[next_id] = intersect / (len(j_neighborhood) + len(k_neighborhood) - intersect)
                next_id += 1
    return row, col, data


def build_mst(similarities, eps=10e-10):
    similarities.data = 1 / similarities.data
    mst = sp.csgraph.minimum_spanning_tree(similarities)
    similarities.data = 1 / similarities.data  # Undo inplace 1/data

    n_cc, cc_labels = sp.csgraph.connected_components(mst, directed=False)
    max_dist = np.max(mst.data)
    min_spanning_tree = np.empty((len(mst.data) + n_cc - 1, 3))
    mst = mst.tocoo()
    min_spanning_tree[: len(mst.data), 0] = mst.row
    min_spanning_tree[: len(mst.data), 1] = mst.col
    min_spanning_tree[: len(mst.data), 2] = mst.data
    if n_cc > 1:
        first_indices = find_first_indices(cc_labels, n_cc)
        for i, index in enumerate(first_indices[1:]):
            min_spanning_tree[len(mst.data) + i, 0] = first_indices[0]
            min_spanning_tree[len(mst.data) + i, 1] = index
            min_spanning_tree[len(mst.data) + i, 2] = max_dist + eps
    return min_spanning_tree


def lc_hslc(adjacency, ms=15):
    triu_adj = sp.triu(adjacency).tocsr()
    row, col, data = _lc_edge_similarity(adjacency.indptr, adjacency.indices, triu_adj.indptr, triu_adj.indices)
    linegraph = sp.coo_matrix((data, (row, col)), shape=(len(triu_adj.data), len(triu_adj.data))).tocsr()
    

    mst = build_mst(linegraph)
    (
        edge_clustering,
        probabilities,
        linkage_tree,
        condensed_tree,
        sorted_mst,
    ) = clusters_from_spanning_tree(
        mst,
        min_cluster_size=ms,
    )

    clustering = edge_clustering_to_node_clustering(adjacency, edge_clustering)
    return clustering


##################
# EGOSPLIT + CAS #
##################

def egosplit_cas(adjacency, tau=0.5):
    egosplit_clusters = EgoSplit(local_clustering="PC", global_clustering=ECG()).fit_predict(adjacency)
    cas_clusters = CASPostProcesser(threshold=0.5, only_remove=False).fit_predict(egosplit_clusters, adjacency)
    return cas_clusters


#########
# OSLOM #
#########

def oslom(adj, oslom_path):
    triu_coo = sp.triu(adj)
    edgelist = np.vstack([triu_coo.row, triu_coo.col]).transpose()
    np.savetxt(f"temp_edgelist.txt", edgelist, fmt="%d")

    cmd = oslom_path+" -f temp_edgelist.txt -uw -singlet -hr 0 -seed 123"
    os.system(cmd+' >/dev/null 2>&1')

    coms = []
    with open(f"temp_edgelist.txt_oslo_files/tp", "r") as f:
        i = 1
        for l in f:
            if i%2 == 0:
                c = np.array(l.split(), dtype="int64")
                if len(c) > 1:
                    coms.append(c)
            i+=1
    oslom = sp.lil_matrix((len(coms), adj.shape[1]), dtype="bool")
    for i, c in enumerate(coms):
        oslom[i, c] = True
    oslom = oslom.tocsr()
    
    os.system('rm temp_edgelist.txt')
    os.system('rm -rf temp_edgelist.txt_oslo_files')
    os.system('rm tp')

    return oslom