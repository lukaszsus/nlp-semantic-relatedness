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
        self.synset_to_vertex_id = None
        # self.max_depth = None   # used for Leacock and Chodorow measure

    def create_graph(self, is_polish=True, relation_types=["hiponimia", "hiperonimia"]):
        """
        It takes long time. Do it once, save model to file and next time load model binary.

        Graph based on wordnet has a structure:
        - all synsets have central vertex,
        - all lexical units in synset are connected to central synset vertex,
        - central synset vertices are connected using relation types specified in relation_types.
        :param is_polish: filter only polish words
        :param relation_types: synset relation types to filter
        :return:
        """
        wn = plwn.load_default()
        self.lemma_to_vertex_id = dict()
        self.synset_to_vertex_id = dict()
        id = 0
        self.g = graph_tool.Graph()
        synset_relation_edges = wn.synset_relation_edges()  # load all synsets relations
        for sr_edge in synset_relation_edges:
            if relation_types is not None and sr_edge.relation.name not in relation_types:
                continue
            if is_polish and (not sr_edge.source.is_polish or not sr_edge.target.is_polish):
                continue
            src_synset_id = sr_edge.source.id
            target_synset_id = sr_edge.target.id
            self._add_synset_vertices(sr_edge, src_synset_id)
            self._add_synset_vertices(sr_edge, target_synset_id)

            # add edge between synsets
            v1 = self.g.vertex(self.synset_to_vertex_id[src_synset_id])
            v2 = self.g.vertex(self.synset_to_vertex_id[target_synset_id])
            self.g.add_edge(v1, v2)

    def _add_synset_vertices(self, sr_edge, synset_id):
        if synset_id not in self.synset_to_vertex_id.keys():
            # adding synsets central vertex
            v_central = self.g.add_vertex()
            self.synset_to_vertex_id[synset_id] = int(v_central)

            # adding edges between lexical units and synset central vertex
            for lexical_unit in sr_edge.source.lexical_units:
                if lexical_unit.lemma not in self.lemma_to_vertex_id.keys():
                    v = self.g.add_vertex()
                    self.lemma_to_vertex_id[lexical_unit.lemma] = int(v)
                else:
                    v = self.g.vertex(self.lemma_to_vertex_id[lexical_unit.lemma])
                self.g.add_edge(v, v_central)

    def save(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name + ".bin"))
        with open(file_path, 'wb') as file:
            pickle.dump(self.g, file)
            pickle.dump(self.lemma_to_vertex_id, file)
            pickle.dump(self.synset_to_vertex_id, file)

    def load(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name))
        if not os.path.isfile(file_path):
            raise ValueError("File {} does not exists.".format(file_path))
        with open(file_path, 'rb') as file:
            self.g = pickle.load(file)
            self.lemma_to_vertex_id = pickle.load(file)
            self.synset_to_vertex_id = pickle.load(file)
            self.g.set_directed(False)

    def semantic_relatedness(self, word1, word2, dist_type='cosine'):
        dist_fun = self.__get_distance_function(dist_type)
        return dist_fun(word1, word2)

    def synonyms(self, word, top=10, dist_type='LeacockChodorow', threshold=100):
        """
        Finds closest top synonyms.
        :param word: search synonyms for that
        :param top: top nearest synonyms
        :param dist_type: distance measure
        :param threshold: we filter candidates for synonyms with shortest path shorter that threshold
        :return: list of words
        """
        # finding shortest paths
        vertex = self.g.vertex(self.lemma_to_vertex_id[word])
        shortest_path = graph_tool.shortest_distance(self.g, source=vertex)
        distances = shortest_path.a
        vertices = list(range(len(shortest_path.a)))

        # sorting vertices by distance
        vertices = np.array(vertices)
        distances = np.array(distances)
        inds = distances.argsort()
        sorted_vertices = vertices[inds]

        # find words which are candidates for nearest neighbours
        # we choose only vertices with shortest path shorter than threshold and lexical (not synsets) vertices
        lemma_ids = list(self.lemma_to_vertex_id.values())
        candidates = list()
        for v_id in sorted_vertices[1:]:        # first one with distance equal to 0 is word vertex
            if v_id in lemma_ids and distances[v_id] < threshold:
                candidates.append(list(self.lemma_to_vertex_id.keys())[list(self.lemma_to_vertex_id.values()).index(v_id)])
                if len(candidates) == top:
                    break

        # sorting with distance measure
        # this part is only a synthetic sugar since we use shortest as the only variable in searching for synonyms
        distances = {}
        dist_fun = self.__get_distance_function(dist_type)
        for neighbour in candidates:
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

    def _LeacockChodorow(self):
        """
        Leacock and Chodorow similarity measure. I have assumed that taxonomy depth is equal to pseudo diameter
        but it is not true.
        :param word1: 
        :param word2: 
        :return: 
        """
        self.depth, (id1, id2) = graph_tool.pseudo_diameter(self.g)

        def distance(word1, word2):
            v1 = self.g.vertex(self.lemma_to_vertex_id.get(word1))
            v2 = self.g.vertex(self.lemma_to_vertex_id.get(word2))
            # graph_tool.shortest_path(g, v1, v2) returns 2 element tuple:
            # list of vertices
            # list of edges
            # we count number of edges
            shortest_path = len(list(graph_tool.shortest_path(self.g, v1, v2)[1]))
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

