import os
import pickle
import numpy as np
import plwn
import graph_tool.all as graph_tool
# install graph-tool instruction: https://git.skewed.de/count0/graph-tool/wikis/installation-instructions
# I have installed it by adding two lines to /etc/apt/sources.list:
# deb http://downloads.skewed.de/apt/bionic bionic main             (for Ubuntu Bionic!)
# deb-src http://downloads.skewed.de/apt/bionic bionic main         (for Ubuntu Bionic!)
# next, do:
# sudo apt-key adv --keyserver keys.openpgp.org --recv-key 612DEFB798507F25
# sudo apt-get update
# sudo apt-get install python-graph-tool
# and it should work
from tqdm import tqdm

from settings import DATA_PATH


class WordNetModel():
    """
    Model using PolishWordNet called 'Slowosiec'.
    """
    def __init__(self):
        self.g = None
        self.lemma_to_vertex_id = None
        # self.max_depth = None   # used for Leacock and Chodorow measure

    def create_graph(self):
        """
        It takes long time. Do it once, save model to file and next time load model binary.
        :return:
        """
        wn = plwn.load_default()
        # TODO choose relevant relations types
        relation_types = ["kolokacyjność"]
        self.lemma_to_vertex_id = dict()
        id = 0
        self.g = graph_tool.Graph()
        lexical_relation_edges = wn.lexical_relation_edges()  # load all lexical relations
        for l_rel_edge in lexical_relation_edges:
            src = l_rel_edge.source
            src_lemma = src.lemma
            target = l_rel_edge.target
            target_lemma = target.lemma
            rel = l_rel_edge.relation
            rel_name = rel.name
            # print(src_lemma + ";", target_lemma + ";", rel_name + ";", rel.aliases)
            if rel_name in relation_types:
                if src_lemma not in self.lemma_to_vertex_id.keys():
                    self.lemma_to_vertex_id[src_lemma] = id
                    id = id + 1
                    self.g.add_vertex()
                if target_lemma not in self.lemma_to_vertex_id.keys():
                    self.lemma_to_vertex_id[target_lemma] = id
                    id = id + 1
                    self.g.add_vertex()
                v1 = self.g.vertex(self.lemma_to_vertex_id[src_lemma])
                v2 = self.g.vertex(self.lemma_to_vertex_id[target_lemma])
                self.g.add_edge(v1, v2)
        self.g.set_directed(False)

    def save(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name + ".bin"))
        with open(file_path, 'wb') as file:
            pickle.dump(self.g, file)
            pickle.dump(self.lemma_to_vertex_id, file)

    def load(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name))
        if not os.path.isfile(file_path):
            raise ValueError("File {} does not exists.".format(file_path))
        with open(file_path, 'rb') as file:
            self.g = pickle.load(file)
            self.lemma_to_vertex_id = pickle.load(file)
            self.g.set_directed(False)

    def semantic_relatedness(self, word1, word2, dist_type='cosine'):
        dist_fun = self.__get_distance_function(dist_type)
        return dist_fun(word1, word2)

    def synonyms(self, word, top=10, dist_type='LeacockChodorow'):
        dist_fun = self.__get_distance_function(dist_type)
        distances = {}
        for neighbour in tqdm(self.lemma_to_vertex_id.keys()):
            distances[neighbour] = dist_fun(word, neighbour)
        sorted_dist = sorted(distances.items(), key=lambda kv: kv[1])
        closest = [sd[0] for sd in sorted_dist[:top]]
        return closest

    def __get_distance_function(self, dist_type='LeacockChodorow'):
        """
        Measures explained in https://arxiv.org/pdf/1310.8059.pdf.
        :param dist_type:
        :return:
        """
        if dist_type == "LeacockChodorow":
            return self._LeacockChodorow()
        elif dist_type == " WuPalmer":
            return self._WuPalmer()

    # def _count_max_depth(self):
    #     """
    #     I am not sure that I understand it correctly but I assume that max depth is longest shortest path
    #     between to vertices.
    #     :return:
    #     """
    #     max_depth = -1
    #     max_int = 2147483647
    #     print("Number of vertices: {}".format(len(list(self.g.vertices()))))
    #     for v in tqdm(self.g.vertices()):
    #         shortest_distances = graph_tool.shortest_distance(self.g, source=v)
    #         shortest_distances = list(filter(lambda x: x != max_int, shortest_distances))
    #         depth = max(shortest_distances)
    #         if depth > max_depth:
    #             max_depth = depth
    #     self.max_depth = max_depth

    def _LeacockChodorow(self):
        """
        TODO
        Leacock and Chodorow similarity measure. I have assumed that taxonomy depth is equal to pseudo diameter.
        :param word1: 
        :param word2: 
        :return: 
        """
        self.depth, (id1, id2) = graph_tool.pseudo_diameter(self.g)

        def distance(word1, word2):
            v1 = self.g.vertex(self.lemma_to_vertex_id.get(word1))
            v2 = self.g.vertex(self.lemma_to_vertex_id.get(word2))
            shortest_path = len(list(graph_tool.shortest_path(self.g, v1, v2)))
            lc_dist = - np.log(shortest_path / (2 * self.depth))
            return lc_dist
        return distance

    def _WuPalmer(self):
        """
        TODO
        :return:
        """
        def distance(word1, word2):
            pass

        return distance

