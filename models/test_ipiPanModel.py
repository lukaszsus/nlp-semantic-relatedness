import time
import numpy as np
from unittest import TestCase

from models.ipi_pan_model import IpiPanModel


class TestIpiPanModel(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestIpiPanModel, self).__init__(*args, **kwargs)
        self.model = IpiPanModel("nkjp+wiki-forms-restricted-100-cbow-hs.txt")

    def test_get(self):
        embedding = self.model.get("krzesło")
        self.assertTrue(embedding is not None)

    def test_semantic_relatedness(self):
        distance_euc = self.model.semantic_relatedness("piec", "piecyk", "euclidean")
        print(distance_euc)
        self.assertGreater(distance_euc, 0)

        distance_man = self.model.semantic_relatedness("piec", "piecyk", "manhattan")
        print(distance_man)
        self.assertGreater(distance_man, 0)

        distance_cos = self.model.semantic_relatedness("piec", "piecyk", "euclidean")
        print(distance_cos)
        self.assertGreater(distance_cos, 0)

    def test_synonyms(self):
        synonyms1 = self.model.synonyms("krzesło")
        synonyms2 = self.model.synonyms("piec")
        synonyms3 = self.model.synonyms("król")

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

    def test_ipimodel(self):
        print(self.model.contains("łódź"))
        print(self.model.contains("Łódźstock"))
        print(self.model.contains("Łódź"))

    def test_binary_raw_loading_time_and_works_the_same(self):
        raw_load_time = time.time()
        model = IpiPanModel("wiki-forms-all-100-cbow-hs.txt")
        raw_load_time = time.time() - raw_load_time
        print("Loading raw took {}".format(raw_load_time))

        start = time.time()
        synonyms_raw = model.synonyms("piec")
        print(synonyms_raw)
        print("Synonyms took {}".format(time.time() - start))

        bin_load_time = time.time()
        model = IpiPanModel("wiki-forms-all-100-cbow-hs.bin")
        bin_load_time = time.time() - bin_load_time
        print("Loading binary took {}".format(bin_load_time))

        start = time.time()
        synonyms_bin = model.synonyms("piec")
        print(synonyms_bin)
        print("Synonyms took {}".format(time.time() - start))

        self.assertGreater(raw_load_time, bin_load_time)
        self.assertTrue(np.sum(np.array(synonyms_raw) == np.array(synonyms_bin))
                        == len(synonyms_raw))
