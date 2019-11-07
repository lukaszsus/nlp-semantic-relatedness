import os
from unittest import TestCase
from datetime import datetime

from models.wordnet_model import WordNetModel
from settings import DATA_PATH


class TestWordNetModel(TestCase):
    def test_create_graph(self):
        wordnet = WordNetModel()
        wordnet.create_graph()
        n_vertices = len(list(wordnet.g.vertices()))
        n_edges = len(list(wordnet.g.edges()))
        self.assertGreater(n_vertices, 0)
        self.assertGreater(n_edges, 0)

    def test_save(self):
        """
        Used to create binary version of graph.
        :return:
        """
        wordnet = WordNetModel()
        wordnet.create_graph(is_polish=False)
        # file_name = "slowosiec-graph-{}".format(datetime.now().strftime("%Y-%m-%d-t%H-%M"))
        file_name = "slowosiec-graph-hiponim"
        wordnet.save(file_name)
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name + ".bin"))
        self.assertTrue(os.path.isfile(file_path))

    def test_load(self):
        wordnet = WordNetModel()
        wordnet.load("slowosiec-graph-polish.bin")
        n_vertices = len(list(wordnet.g.vertices()))
        n_edges = len(list(wordnet.g.edges()))
        self.assertGreater(n_vertices, 0)
        self.assertGreater(n_edges, 0)

    # def test__count_max_depth(self):
    #     wordnet = WordNetModel()
    #     wordnet.load("slowosiec-graph-2019-11-06-t22-14.bin")
    #     wordnet._count_max_depth()
    #     print(wordnet.max_depth)
    #     self.assertGreater(wordnet.max_depth, 0)

    def test__LeacockChodorow(self):
        wordnet = WordNetModel()
        # wordnet.load("slowosiec-graph-2019-11-07-t17-45.bin")
        wordnet.load("slowosiec-graph-hiponim.bin")
        dist = wordnet._LeacockChodorow()
        print(dist("bastion", "zamek"))
        print(dist("piec", "piekarnik"))
        print(dist("król", "porzeczka"))
        print(dist("król", "królowa"))
        self.assertGreater(dist("piec", "piekarnik"), 0)
        self.assertGreater(dist("król", "królowa"), dist("król", "porzeczka"))

    def test_synonyms(self):
        """
        Best are options:
        - filtering polish and all types of relations (slowosiec-graph-polish)
        - not filtering polish and hiperonim/hiponim relation type (slowosiec-graph-hiponim)
        :return:
        """
        wordnet = WordNetModel()
        # wordnet.load("slowosiec-graph-2019-11-07-t17-45.bin")
        wordnet.load("slowosiec-graph-hiponim.bin")

        synonyms1 = wordnet.synonyms("krzesło")
        synonyms2 = wordnet.synonyms("piec")
        synonyms3 = wordnet.synonyms("król")

        print(synonyms1)
        print(synonyms2)
        print(synonyms3)

        self.assertTrue(synonyms1 is not None)
        self.assertTrue(synonyms2 is not None)
        self.assertTrue(synonyms3 is not None)
        self.assertTrue(type(synonyms1) == list)
        self.assertTrue(type(synonyms2) == list)
        self.assertTrue(type(synonyms3) == list)
        self.assertTrue(type(synonyms1[0]) == str)
        self.assertTrue(type(synonyms2[0]) == str)
        self.assertTrue(type(synonyms3[0]) == str)
