from collections import defaultdict, Counter
from sklearn.metrics import homogeneity_completeness_v_measure
import numpy as np
import pandas as pd
# import seaborn as sb
import matplotlib.pyplot as pp
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics import adjusted_rand_score
from numpy.random import choice
import networkx as nx
from multiprocessing import Pool, cpu_count
import dill
import community
import time
import bisect as bs
from heapq import nlargest
from operator import itemgetter

def run_dill_encoded(payload):
    fun, args = dill.loads(payload)
    return fun(*args)


def apply_async(pool, fun, args):
    payload = dill.dumps((fun, args))
    return pool.apply_async(run_dill_encoded, (payload,))


class network_model:
    def __init__(self, alpha, gamma, tau, thrsh, K_0, net, dir, iters, w_0, prev_model, com_prec):
        """
        :param alpha:
        :param gamma:
        :param tau:
        :param thrsh: defines the convergence threshold
        :param K_0: initial number of clusters
        :param net: graph in networkx
        :param dir: directory of files
        :param iters: number of iterations
        :param w_0: initial weights over edges
        :param seed: seed
        :param prev_model: for dynamic setting, previous step model
        """
        print("*** NetWork Model Class ***")
        self.alpha, self.alpha_orig, self.gamma, self.tau = alpha, alpha, gamma, tau
        self.thrsh, self.num_iters, self.dir = thrsh, iters, dir
        self.K, self.weights = K_0, w_0
        self.graph = net
        self.prev_model = prev_model
        np.random.seed(20)

        """data points"""
        self.links = list(net.edges())
        self.nodes = list(net.nodes())
        self.p, self.p_final = dict(), defaultdict(lambda: 0.0)

        """for clustering settings"""
        self.c = dict()  # cluster assignment of each item
        self.clusters = defaultdict(set)  # partitioning of components

        self.eta = []  # number of edges in each cluster
        self.l_out = dict()  # number of in-links of each node for each cluster
        self.l_in = dict()  # number of out-links of each node for each cluster
        self.betas_out, self.betas_in = dict(),dict()
        self.betas_n_out, self.betas_n_in =[],[]

        self.c_nodes = dict()
        self.community_detection_accuracy = 0
        self.expected_community_accuracy = com_prec

        # best partitioning
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        G.add_edges_from(self.links)
        partitions = community.best_partition(G)
        to_remove = []
        for n in partitions.keys():
            if n not in self.nodes:
                to_remove.append(n)
        for n in to_remove:
            partitions.pop(n)
        self.partition = partitions


    def initialize_clusters_orig(self, first_network):
        K_part = max(self.partition.values()) + 1
        # print("~~ Initializing network ... prev model is None:", first_network," K:",K_part)
        # assigning links to clusters
        # if randomly assign run this line
        # self.c = {e: np.random.choice(self.K) for e in self.links}
        # else run the following if else
        if first_network:
            for (u, v) in self.links:
                if u not in self.partition.keys():
                    if v not in self.partition.key():
                        self.c[(u, v)] = 0
                    else:
                        self.c[(u, v)] = self.partition[v]
                else:
                    self.c[(u, v)] = self.partition[u]
                self.clusters[self.c[(u,v)]].add((u,v))
            self.K = K_part
            # self.eta = [0 for _ in range(self.K)]  # number of edges in each cluster
            # self.l_out = defaultdict(lambda: [0 for _ in range(self.K)])  # number of in-links of each node for each cluster
            # self.l_in = defaultdict(lambda: [0 for _ in range(self.K)])  # number of out-links of each node for each cluster
            # self.betas_out, self.betas_in = defaultdict(lambda: [0 for _ in range(self.K)]), defaultdict(
            #     lambda: [0 for _ in range(self.K)])
            # self.betas_n_out, self.betas_n_in = [0 for _ in range(self.K)], [0 for _ in range(self.K)]
        else:
            """Reading from previous iterations model"""
            # self.c, self.c_nodes, self.betas_n_out, self.betas_n_in, self.betas_out, self.betas_in, \
            # self.l_out, self.l_in, self.K = self.prev_model
            self.c, self.c_nodes, self.K = self.prev_model
            self.K = max(max(self.c.values()) + 1, K_part)
            """intitalize variables for new nodes"""
            # self.eta = [0 for _ in range(self.K)]
            # if K_part > self.K:
            #     # self.eta += [0 for _ in range(self.K, K_part)]
            #     for u in self.nodes:
            #         self.l_out[u] += [0 for _ in range(self.K, K_part)]
            #         self.l_in[u] += [0 for _ in range(self.K, K_part)]
            #         self.betas_out[u] += [0 for _ in range(self.K, K_part)]
            #         self.betas_in[u] += [0 for _ in range(self.K, K_part)]
            #     self.betas_n_out += [0 for _ in range(self.K, K_part)]
            #     self.betas_n_in += [0 for _ in range(self.K, K_part)]
            # else:
            #     K_part = self.K
            """initialize cluster for new edges"""
            for link in self.links:
                if link in self.c:
                    self.clusters[self.c[link]].add(link)
                    continue
                if link[0] in self.c_nodes:
                    self.c[link] = self.c_nodes[link[0]]
                elif link[1] in self.c_nodes:
                    self.c[link] = self.c_nodes[link[1]]
                else:
                    self.c[link] = self.partition[link[0]]  # np.random.randint(0, self.K)
                    # k = self.c[link]
                    # self.l_out[link[0]][k] += 1
                    # self.l_in[link[1]][k] += 1
                self.clusters[self.c[link]].add(link)
            # self.sample_betas()  # TODO

            # for node in self.nodes:
            #     self.c_nodes(node)
            # self.K = K_part

        # print("~~~~clusters ",self.clusters)
        self.eta = [0 for _ in range(self.K)]  # number of edges in each cluster
        self.l_out = defaultdict(lambda: [0 for _ in range(self.K)])  # number of in-links of each node for each cluster
        self.l_in = defaultdict(lambda: [0 for _ in range(self.K)])  # number of out-links of each node for each cluster
        self.betas_out, self.betas_in = defaultdict(lambda: [0 for _ in range(self.K)]), defaultdict(
            lambda: [0 for _ in range(self.K)])
        self.betas_n_out, self.betas_n_in = [0 for _ in range(self.K)], [0 for _ in range(self.K)]
        """after initialization update variables"""
        """remove empty clusters"""
        for (k, count) in Counter(self.c.values()).items():
            self.eta[k] = count
        # print("self.eta after initialization: ",self.eta)
        while 0 in self.eta:
            # print("0 in self.eta")
            k_to_del = self.eta.index(0)
            self.remove_cluster(k_to_del)
        self.alpha = self.alpha_orig / self.K
        for e in self.links:
            k = self.c[e]
            u, v = e
            self.l_out[u][k] += 1
            self.l_in[v][k] += 1
        for node in self.nodes:
            self.cluster_assignment_of_nodes(node)
        for k in range(self.K):
            assert self.eta[k]>0 ,'cluster %d has zero elements'%k
        

    def initialize_clusters(self, first_network):
        print("~~ Initializing network ... prev model is None:", first_network)
        if first_network:
            self.c_nodes = self.partition
            for (u, v) in self.links:
                if u not in self.partition.keys():
                    if v not in self.partition.key():
                        self.c[(u, v)] = 0
                        self.c_nodes[u] = 0
                        self.c_nodes[v] = 0
                    else:
                        self.c[(u, v)] = self.partition[v]
                else:
                    self.c[(u, v)] = self.partition[u]
                self.clusters[self.c[(u, v)]].add((u, v))
        else:
            """Reading from previous iterations model"""
            self.c, self.c_nodes, self.K = self.prev_model
            """initialize cluster for new edges"""
            for link in self.links:
                if link in self.c:
                    self.clusters[self.c[link]].add(link)
                    continue
                if link[0] in self.c_nodes:
                    self.c[link] = self.c_nodes[link[0]]
                elif link[1] in self.c_nodes:
                    self.c[link] = self.c_nodes[link[1]]
                else:
                    self.c[link] = self.partition[link[0]]  # np.random.randint(0, self.K)
                self.clusters[self.c[link]].add(link)
        self.K = max(self.c.values()) + 1
        self.eta = [0 for _ in range(self.K)]  # number of edges in each cluster
        for node in self.nodes:
            self.l_out[node] = [0 for _ in range(self.K)]  # number of in-links of each node for each cluster
            self.l_in[node] = [0 for _ in range(self.K)]  # number of out-links of each node for each cluster
            self.betas_out[node] = [0 for _ in range(self.K)]
            self.betas_in[node] = [0 for _ in range(self.K)]
        self.betas_n_out, self.betas_n_in = [0 for _ in range(self.K)], [0 for _ in range(self.K)]
        """after initialization update variables"""
        for (k, count) in Counter(self.c.values()).items():
            self.eta[k] = count
        """remove empty clusters"""
        while 0 in self.eta:
            # print("0 in self.eta")
            k_to_del = self.eta.index(0)
            self.remove_cluster(k_to_del)
        for k in range(self.K):
            assert self.eta[k] > 0, 'cluster %d has zero elements' % k

        for e in self.links:
            u, v = e
            k = self.c[e]
            self.l_out[u][k] += 1
            self.l_in[v][k] += 1

        self.alpha = self.alpha_orig / self.K
        self.sample_betas()
        for node in self.nodes:
            self.cluster_assignment_of_nodes(node)

    def inference(self,parallel):
        """
        Gibbs sampler
        here we make use of split/merge gibbs sampling
        :return:converges when reaching the threshold
        """
        self.initialize_clusters(self.prev_model is None)  # TODO self.prev_mode to self.prev_mode[0]
        edges = list(set(self.links))
        num_edges = len(edges)
        iterations = 0
       

        def monte_carlo_part(i, e_s):
            print("monte carlo", i, e_s)
            for e in e_s:
                print("edge is processing.. ",e)
                self.sample_betas()
                self.sample_cluster_assignment(e)
           

        while iterations < self.num_iters:
           
            prev_cluster = [self.c[e] for e in edges]
            num_process = cpu_count()
            part_count = int(num_edges / num_process)  # for _ in range(num_process)]
            if part_count != num_edges / num_process:
                part_count += 1
            results = []
            s_t=time.time()
            if parallel:  # num_process<num_edges
                pool = Pool(processes=num_process)
                # result = pool.map(func=sample_edges,args=(edges,self))
                for i in range(num_process):
                    if (i + 1) * part_count > num_edges:
                        e_list = edges[i * part_count:]
                    else:
                        e_list = edges[i * part_count:(i + 1) * part_count]
                    if len(e_list) == 0: break
                    results.append(apply_async(pool, monte_carlo_part, (i, e_list)))
                results = [p.get() for p in results]

            else:
                for e in edges:
                    self.sample_betas()
                    self.sample_cluster_assignment(e)
            # print("******** edges done in ****** ",time.time()-s_t)
            changed_perc = 1 - adjusted_rand_score(prev_cluster, [self.c[e] for e in edges])
            self.community_similarity()
            if changed_perc >= self.thrsh or len(edges) < 3:
                # print("breaked!", changed_perc, iterations, self.community_detection_accuracy)
                break
            iterations += 1

          
    def community_similarity(self):
        true_vals = [self.partition[node] for node in sorted(self.nodes)]
        pred_vals = [self.c_nodes[node] for node in sorted(self.nodes)]
        self.community_detection_accuracy = normalized_mutual_info_score(true_vals, pred_vals)

    def sample_cluster_assignment(self, link):
        """
        Given u, v, other cluster assignments and Betas
        :param link: sender, receiver
        :return:
        """
        old_k = self.c[link]

        self.remove_from_cluster(link)
        new_k = self.cluster_posterior(link)
        # print("Assigning link %s from %d to %d .." % (link, self.c[link], new_k))

        if new_k == self.K:
            new_k = self.add_new_cluster()

        self.assign_to_cluster(link, new_k)

        if self.eta[old_k] == 0:
            self.remove_cluster(old_k)
        nodes_to_change = set(sum(self.clusters[new_k] | self.clusters[old_k], ()))
        for u in nodes_to_change:
            self.cluster_assignment_of_nodes(u)

    def remove_from_cluster(self, link):
        """
        remove link from its current cluster, if number of links=0 remove cluster
        :param link:
        """
        k = self.c[link]  # cluster assignment of link
        # print("removing link %s from %d with cluster:%s" % (link, k, self.clusters[k]))
        u, v = link
        # decrease counters
        self.clusters[k].remove(link)
        self.l_out[u][k] -= 1
        self.l_in[v][k] -= 1
        self.eta[k] -= 1

    def add_new_cluster(self):
        """
        :return: cluster id of new cluster that has been added
        """
        k_new = self.K
        self.eta.append(0)

        for i in self.nodes:
            self.l_in[i].append(0)
            self.l_out[i].append(0)
            self.betas_out[i].append(0)
            self.betas_in[i].append(0)
        self.betas_n_out.append(0)
        self.betas_n_in.append(0)

        self.K += 1
        self.alpha = self.alpha_orig / self.K
        return k_new

    def assign_to_cluster(self, link, new_k):
        """
        assigns link to cluster k_new and updates parameters
        :param link:
        :param new_k:
        """
        self.c[link] = new_k

        self.eta[new_k] += 1
        self.clusters[new_k].add(link)

        u, v = link

        self.l_out[u][new_k] += 1
        self.l_in[v][new_k] += 1

    def cluster_posterior(self, link):
        u, v = link
        theta = [0] * self.K
        cluster_crps = np.array(self.eta) / (sum(self.eta) + self.alpha)
        cluster_crps[-1] = self.alpha / (sum(self.eta) + self.alpha)
        new_k = 0
        for k in range(self.K):
            if self.l_out[u][k] != 0 and self.l_in[v][k] != 0:
                theta[k] = (self.eta[k] / (sum(self.eta) + self.alpha)) * \
                           ((self.l_out[u][k] + (self.tau * self.betas_out[u][k])) / (self.tau + self.eta[k])) * \
                           ((self.l_in[v][k] + (self.tau * self.betas_in[v][k])) / (self.tau + self.eta[k]))
                new_k = (self.alpha / (sum(self.eta) + self.alpha)) * self.betas_out[u][k] * self.betas_in[v][k]
            elif self.l_out[u][k] != 0 and self.l_in[u][k] == 0:
                theta[k] = (self.eta[k] / (sum(self.eta) + self.alpha)) * \
                           ((self.l_out[u][k] + (self.tau * self.betas_out[u][k])) / (self.tau + self.eta[k])) * \
                           (self.betas_n_in[k])
                new_k = (self.alpha / (sum(self.eta) + self.alpha)) * self.betas_out[u][k] * self.betas_n_in[k]
            elif self.l_out[u][k] == 0 and self.l_in[u][k] != 0:
                theta[k] = (self.eta[k] / (sum(self.eta) + self.alpha)) * \
                           (self.betas_n_out[k]) * \
                           ((self.l_in[v][k] + (self.tau * self.betas_in[v][k])) / (self.tau + self.eta[k]))
                new_k = (self.alpha / (sum(self.eta) + self.alpha)) * self.betas_n_out[k] * self.betas_in[v][k]
            else:
                theta[k] =(self.eta[k] / (sum(self.eta) + self.alpha)) * \
                          self.betas_n_out[k] * self.betas_n_in[k]
                new_k = (self.alpha / (sum(self.eta) + self.alpha))*\
                        self.betas_n_out[k] * self.betas_n_in[k]
        theta.append(new_k)
        max_indices = np.argwhere(np.max(theta) == theta)
        if len(max_indices) == 1:
            max_index= max_indices[0][0]
        else:
            idx = np.random.choice(len(max_indices))
            max_index= max_indices[idx][0]
        if link not in self.weights:
            self.p[link] = round(theta[max_index], 6)
        else:
            self.p[link] = round(theta[max_index] * self.weights[link],6)
        return max_index

    def sample_betas(self):
        def f(x):
            if x == 0: return 1
            return x
        n_s_nodes_with_zero_links = [0 for _ in range(self.K)]
        n_r_nodes_with_zero_links = [0 for _ in range(self.K)]
        for k in range(self.K):
            for n in self.nodes:
                if self.l_out[n][k] >0:
                    self.betas_out[n][k] = (self.l_out[n][k]) / (self.gamma + self.eta[k])
                    n_s_nodes_with_zero_links[k]+=1
                else:
                    self.betas_out[n][k]=0
                if self.l_in[n][k] >0:
                    self.betas_in[n][k] = (self.l_in[n][k]) / (self.gamma + self.eta[k])
                    n_r_nodes_with_zero_links[k]+=1
                else:
                    self.betas_out[n][k]=0.0
        # n_s_nodes_with_zero_links = list(map(lambda x: 1 if x == 0 else x, n_s_nodes_with_zero_links))
        n_s_nodes_with_zero_links = list(map(f, n_s_nodes_with_zero_links))
        n_r_nodes_with_zero_links = list(map(f, n_r_nodes_with_zero_links))
        self.betas_n_out = [self.gamma / (n_s_nodes_with_zero_links[k] *(self.gamma + self.eta[k])) for k in range(self.K)] #np.sum(list(self.l_out.values())) + np.sum(list(self.l_in.values()
        self.betas_n_in = [self.gamma / (n_r_nodes_with_zero_links[k] *(self.gamma + self.eta[k])) for k in range(self.K)] 

    def remove_cluster(self, k):
        """
        when a cluster is removed, the corresponding parameters of this cluster should be removed
        :param k: index of cluster
        """
        # print("Removing cluster %d with items %s" % (k, self.clusters[k]))
        for cluster in range(self.K):
            if cluster > k:
                for link in self.clusters[cluster]:
                    self.c[link] -= 1
                self.clusters[cluster - 1] = self.clusters.pop(cluster)

        # print("to remove k:",k," self.eta:",self.eta, len(self.eta))
        self.K -= 1
        self.eta.pop(k)
        # print(k," removed ===> ",[len(self.l_out[node]) for node in self.nodes])
        for node in self.nodes:
            # print("----->",node,len(self.l_out[node]))
            self.l_out[node].pop(k)
            self.l_in[node].pop(k)
            self.betas_out[node].pop(k)
            self.betas_in[node].pop(k)
        self.betas_n_out.pop(k)
        self.betas_n_in.pop(k)
        self.alpha = self.alpha_orig / self.K

    def print_info(self, pred_clustering, true_clustering, i):
        print('Iteration: {0}'.format(i))
        print('Number of cluster: {}'.format(len(np.unique(pred_clustering))))
        print('Homogeneity: {0}, Completeness: {1}, V-measure: {2}'.format(
            *homogeneity_completeness_v_measure(pred_clustering, true_clustering))
        )

    def cluster_assignment_of_nodes(self, i):
        betas = [(self.l_in[i][k] + self.l_out[i][k]) / self.eta[k] for k in range(self.K - 1)]
        k_maxes = np.argwhere(betas == np.max(betas)).flatten().tolist()
        if len(k_maxes) == 1:
            self.c_nodes[i] = k_maxes[0]
        elif len(k_maxes) > 1:
            self.c_nodes[i] = choice(k_maxes)
        else:
            self.c_nodes[i] = -1

    def add_node(self, node):
        self.nodes.append(node)
        self.l_out[node] = [0 for _ in range(self.K)]
        self.l_in[node] = [0 for _ in range(self.K)]
        self.betas_out[node], self.betas_in[node] = [0 for _ in range(self.K)], [0 for _ in range(self.K)]
        self.cluster_assignment_of_nodes(node)

    def update_observations(self, cascade, hit_times, V):
        print("******* Updating observations *******", len(self.p),len(V))
        self.p_final.clear()
        mpt = dict()
        nodes_max_node = defaultdict(lambda: [])
        not_possible_edges = defaultdict(int)
        all_edges=set()
        for c_id, propagation in cascade.items():
            for u in V[c_id]:
                for v in V[c_id]:
                    if u == v: continue
                    dt = hit_times[c_id][v] - hit_times[c_id][u]
                    edge = (u, v)
                    if dt > 0:
                        not_possible_edges[edge] = 1
                        all_edges.add(edge)
                        if edge not in self.p.keys():
                            self.cluster_posterior(edge)
                        bs.insort(nodes_max_node[v], (self.p[edge], u))
                    else:
                        not_possible_edges[edge] = 0
            mpt[c_id] = self.find_max_tree(nodes_max_node)
        for edge, p in self.p.items():
            if edge not in self.p_final.keys():
                self.p_final[edge] = p
            elif not_possible_edges[edge] == 0 and edge not in all_edges:
                self.p_final[edge] = 0
        return [inner for outer in mpt.values() for inner in outer]

    def update_observations_N_trees(self, cascade, hit_times, V, to_infer):
        print("******* Updating observations N Trees *******")
        s_time = time.time()
        self.p_final.clear()
        mpt = dict()
        not_possible_edges = dict()
        observations=set()
        link_weights = dict()

        for c_id in cascade.keys():
            link_weights[c_id] = dict()
            not_possible_edges[c_id] = dict()
            nodes_max_node = defaultdict(lambda: [])
            for u in V[c_id]:
                for v in V[c_id]:
                    if u == v: continue
                    dt = hit_times[c_id][v] - hit_times[c_id][u]
                    edge = (u, v)

                    if dt > 0:
                        if edge not in self.p:
                            self.cluster_posterior(edge)
                        not_possible_edges[c_id][edge] = 1
                        bs.insort(nodes_max_node[v], (self.p[edge], u))
                        link_weights[c_id][edge] = self.p[edge]
                    else:
                        link_weights[c_id][edge] = 0
                        not_possible_edges[c_id][edge] = 0
                    # if edge == (587, 648):
                    #     print("c_id, (587, 648):",c_id,link_weights[c_id][edge])
            mpt[c_id] = self.find_max_tree(nodes_max_node)
            observations |= set(mpt[c_id])
        count_casc=0
        while len(observations) < to_infer:
            for c_id in cascade.keys():
                # print("c_id:",c_id)
                new_links, highest_link = self.find_max_N_tree(link_weights[c_id], list(mpt[c_id]))
                if len(new_links) > 0:
                    mpt[c_id].append(highest_link)
                    if (highest_link[1], highest_link[0]) not in observations:
                        observations.add(highest_link)
                count_casc += 1
            if len(observations) >= to_infer or count_casc >= len(cascade):
                break

        not_possible_ = []
        links_all = sorted(self.p, key=self.p.get, reverse=True)
        for link in links_all:
            count_Zero = 0
            count_One = 0
            for i in cascade.keys():
                if link in not_possible_edges[i]:
                    if not_possible_edges[i][link] == 0:
                        count_Zero += 1
                    else:
                        count_One += 1
            if count_Zero > count_One:
                not_possible_.append(link)
        for link in link_weights[c_id].keys():
            if link in not_possible_: self.p_final[link] = 0
            else: self.p_final[link] = self.p[link]

        for link, prob in self.p.items():
            if link not in self.p_final.keys():
                self.p_final[link] = prob
        print("CASCADE DONE in ",time.time()-s_time)
        return [inner for outer in mpt.values() for inner in outer]

    def update_observations_count_zero_1(self, cascade, hit_times, V):
        print("******* Updating observations count zero *******")
        s_time = time.time()
        self.p_final.clear()
        mpt = dict()
        nodes_max_node = defaultdict(lambda: [])
        not_possible_edges = defaultdict(int)
        all_edges = set()
        counttt=0
        counttt_2=0
        links_all = sorted(self.p, key=self.p.get, reverse=True)
        for c_id, propagation in cascade.items():
            not_possible_edges[c_id] = dict()
            for u in V[c_id]:
                for v in V[c_id]:
                    if u == v: continue
                    dt = hit_times[c_id][v] - hit_times[c_id][u]
                    edge = (u, v)
                    counttt += 1
                    if dt > 0:
                        counttt_2 += 1
                        not_possible_edges[c_id][edge] = 1
                        all_edges.add(edge)
                        if edge not in self.p.keys():
                            self.cluster_posterior(edge)
                        bs.insort(nodes_max_node[v], (self.p[edge], u))
                    else:
                        not_possible_edges[c_id][edge] = 0
            mpt[c_id] = self.find_max_tree(nodes_max_node)
        # s_time = time.time()
        # self.p_final.clear()
        # links_all = sorted(self.p, key=self.p.get, reverse=True)
        # mpt = dict()
        # not_possible_edges = dict()
        # link_weights = dict()
        # counttt=0
        # counttt_2=0
        # for c_id in cascade.keys():
        #     link_weights[c_id] = dict()
        #     not_possible_edges[c_id] = dict()
        #     for u in V[c_id]:
        #         for v in V[c_id]:
        #             if u == v: continue
        #             dt = hit_times[c_id][v] - hit_times[c_id][u]
        #             edge = (u, v)
        #             counttt+=1
        #             if dt > 0:
        #                 counttt_2+=1
        #                 if edge not in self.p:
        #                     self.cluster_posterior(edge)
        #                 not_possible_edges[c_id][edge] = 1
        #                 link_weights[c_id][edge] = self.p[edge]
        #             else:
        #                 link_weights[c_id][edge] = 0
        #                 not_possible_edges[c_id][edge] = 0
        #     mpt[c_id] = self.find_max_tree_count_zero(link_weights[c_id], V[c_id])
        print(counttt,counttt_2,"~~!!~~ ",len([inner for outer in mpt.values() for inner in outer]))
        not_possible_ = []
        for link in links_all:
            count_Zero = 0
            count_One = 0
            for i in cascade.keys():
                if link in not_possible_edges[i]:
                    if not_possible_edges[i][link] == 0:
                        count_Zero += 1
                    else:
                        count_One += 1
            if count_Zero > count_One:
                not_possible_.append(link)
        for link in not_possible_: self.p_final[link] = 0
        for link, prob in self.p.items():
            if link not in self.p_final.keys():
                self.p_final[link] = prob
        print("CASCADE DONE in ", time.time() - s_time)
        return [inner for outer in mpt.values() for inner in outer]

    def update_observations_count_zero(self, cascade, hit_times, V):
        print("******* Updating observations count zero *******")
        start_t = time.time()
        self.p_final.clear()
        links_all = sorted(self.p, key=self.p.get, reverse=True)
        mpt = defaultdict(lambda: [])
        test_not_p = dict()
        link_weights = {}
        counttt = 0
        counttt_2 = 0
        for casc_id in cascade.keys():
            test_not_p[casc_id] = dict()
            link_weights[casc_id] = {}
            for s in V[casc_id]:
                for r in V[casc_id]:
                    if s == r: continue
                    dt = hit_times[casc_id][r] - hit_times[casc_id][s]
                    link = (s, r)
                    counttt += 1
                    if dt <= 0:
                        link_weights[casc_id][link] = 0
                        test_not_p[casc_id][link] = 0
                    if dt > 0:
                        counttt_2 += 1
                        if link not in self.p.keys():
                            self.cluster_posterior(link)
                        test_not_p[casc_id][link] = 1
                        link_weights[casc_id][link] = self.p[link]
            mpt[casc_id] = self.find_max_tree_count_zero(link_weights[casc_id], V[casc_id])

        print(counttt, counttt_2, "~~!!~~ ", len([inner for outer in mpt.values() for inner in outer]))

        not_possible_links_for_me = []
        for link in links_all:
            count_Zero = 0
            count_One = 0
            for i in cascade.keys():
                if link in test_not_p[i]:
                    if test_not_p[i][link] == 0:
                        count_Zero += 1
                    else:
                        count_One += 1
            if count_Zero > count_One:
                not_possible_links_for_me.append(link)
        for link in not_possible_links_for_me:
            self.p_final[link] = 0

        for link, prob in self.p.items():
            if link not in self.p_final.keys():
                self.p_final[link] = prob
        print("CASCADES Done ..", time.time() - start_t)
        return [inner for outer in mpt.values() for inner in outer]

    def find_max_tree(self, nodes_max):
        tree = []
        for node, in_nodes in nodes_max.items():
            if len(in_nodes) != 0:
                e = (in_nodes[0][1], node)
                tree.append(e)
                self.p_final[e] = in_nodes[0][0]
        return tree
    def find_max_tree_count_zero(self, w, casc_nodes):
        tree = []
        E = w.keys()
        for i in casc_nodes:
            # if i == roots:
            #     continue
            possible_links = [(j, i) for j in casc_nodes if (j, i) in E]
            if len(possible_links) == 0:
                continue
            index_j = np.argmax([w[possible_links[j]] for j in range(len(possible_links))])
            link = possible_links[index_j]
            tree.append(link)
        for link in E:
            self.p_final[link] = w[link]
        return tree

    def find_max_N_tree(self, w, current_links):
        weights_not_in_current = defaultdict(lambda: [])
        for k, v in w.items():
            if k in current_links:
                continue
            weights_not_in_current[k] = v
        if len(weights_not_in_current) == 0:
            return [], None
        # print("weights_not_in_current",weights_not_in_current)
        (index, highest_w) = nlargest(1, enumerate(weights_not_in_current.values()), itemgetter(1))[0]
        highest_link = list(weights_not_in_current.keys())[index]
        node_to_substitude = highest_link[1]
        for link in current_links:
            if link[1] == node_to_substitude:
                current_links.remove(link)
                # print("link removed in cascade:",link)
        current_links.append(highest_link)
        # print("link added in cascade",highest_link)
        for link in current_links:
            if link not in self.p_final.keys():
                self.p_final[link] = w[link]
            else:
                self.p_final[link] += w[link]
        return current_links, highest_link

