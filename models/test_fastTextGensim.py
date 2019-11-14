from unittest import TestCase

from models.fast_text_gensim import FastTextGensim


class TestIpiPanModelGensim(TestCase):
    def __init__(self, *args, **kwargs):
        super(TestIpiPanModelGensim, self).__init__(*args, **kwargs)
        self.model = FastTextGensim("kgr10.plain.lemma.lower.cbow.dim100.neg10.bin-filtered.bin")

    def test_get(self):
        print(self.model.model.wv["krzesło"])

    def test_semantic_relatedness(self):
        distance_euc = self.model.semantic_relatedness("piec", "piecyk", "euclidean")
        distance_man = self.model.semantic_relatedness("piec", "piecyk", "cosine")
        print(distance_euc)
        print(distance_man)
        self.assertGreater(distance_euc, 0)
        self.assertGreater(distance_man, 0)

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
