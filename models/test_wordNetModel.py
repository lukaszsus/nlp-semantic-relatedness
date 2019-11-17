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
        file_name = "slowosiec-graph-hiperonim-connected-by-top-sort"
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

    def test__LeacockChodorow(self):
        wordnet = WordNetModel()
        wordnet.load("slowosiec-graph-hiperonim-connected-by-top-sort.bin")
        dist = wordnet._LeacockChodorow()
        print(dist("łatwy", "męczący"))

    def test_synonyms(self):
        """
        Best are options:
        - filtering polish and all types of relations (slowosiec-graph-polish)
        - not filtering polish and hiperonim/hiponim relation type (slowosiec-graph-hiponim)
        :return:
        """
        wordnet = WordNetModel()
        wordnet.load("slowosiec-graph-hiperonim-connected-by-top-sort.bin")

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

    def test__WuPalmer(self):
        wordnet = WordNetModel()
        wordnet.load("slowosiec-graph-hiperonim-connected-by-top-sort.bin")
        dist = wordnet._WuPalmer()
        print(dist("łatwy", "męczący"))

    def test_wu_palmer_synonyms(self):
        """
        Best are options:
        - filtering polish and all types of relations (slowosiec-graph-polish)
        - not filtering polish and hiperonim/hiponim relation type (slowosiec-graph-hiponim)
        :return:
        """
        wordnet = WordNetModel()
        wordnet.load("slowosiec-graph-hiperonim-connected-by-top-sort.bin")

        synonyms1 = wordnet.synonyms("krzesło", dist_type="WuPalmer")
        synonyms2 = wordnet.synonyms("piec", dist_type="WuPalmer")
        synonyms3 = wordnet.synonyms("król", dist_type="WuPalmer")

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

    def test__count_root_avg_depth(self):
        wordnet = WordNetModel()
        wordnet.load("slowosiec-graph-hiperonim-connected-by-top-sort.bin")
        depth = wordnet._count_root_depth()
        print(depth)
        self.assertGreater(depth, 0)
