"""
Since 6.11.2019 we do not use gensim so this is deprecated.
It is not deleted because of possibility of returning to this idea.
"""

import os
import numpy as np
import gensim
from deprecated import deprecated
from scipy.spatial import distance
from settings import DATA_PATH
from tqdm import tqdm

@deprecated(reason="Class is not finished because it is hard to convert dict to gensim.")
class IpiPanModelGensim():
    """
    DATA_PATH/models directory should contains saved models.
    Models are available to download on http://dsmodels.nlp.ipipan.waw.pl/.

    In contrast to IpiPanModel, this class requires binary Word2Vec binary models.

    Naming convention from IPI-PAN docs:
        Nazwa pliku: corpus-type-stype-dim-arch-alg.txt.gz
        corpus      nazwa korpusu - nkjp, wiki lub nkjp+wiki
        type        typ modelu - model oparty na formach (forms) lub lematach (lemmas)
        stype       podtyp modelu - wszystkie części mowy (all) lub tylko wybrane części mowy (restricted)
        dim         rozmiar wektora - 100 lub 300
        arch        architektura sieci neuronowej - CBOW (cbow) lub Skip-Gram (skipg)
        alg         algorytm uczący - Hierarchical Softmax (hs) lub Negative Sampling (ns)

        Niektóre modele ograniczone zostały tylko do tych słów, które wystąpiły co najmniej 30 lub 50 razy w korpusie.
        Jest to zaznaczone po nazwie algorytmu uczącego alg. it100 w nazwie pliku oznacza, że dany model został
        wytrenowany w stu iteracjach.
    """
    def __init__(self, file_name):
        self.model = None
        self.load_model(file_name)

    def load_model(self, file_name):
        file_extension = os.path.splitext(file_name)[1]
        if file_extension == '.txt':
            self.load_model_from_raw_file(file_name)
        else:
            self.load_model_from_binary(file_name)

    def load_model_from_binary(self, file_name):
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name))
        if not os.path.isfile(file_path):
            raise ValueError("File {} does not exists.".format(file_path))
        self.model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=True)

    def load_model_from_raw_file(self, file_name):
        self.load_raw_file(file_name)
        # embeddings = self.filter_model_with_polimorf(embeddings)
        print(len(self.word2vec_embeddings.keys()))
        self.model = gensim.models.keyedvectors.Word2VecKeyedVectors(len(self.word2vec_embeddings.keys()))
        self.model.vocab = self.word2vec_embeddings
        vectors = np.array(self.word2vec_embeddings.values())
        print(type(vectors))
        print(vectors.shape)
        self.model.vectors = vectors

    def load_raw_file(self, file_name):
        self.word2vec_embeddings = dict()
        file_path = os.path.join(DATA_PATH, os.path.join("models", file_name))
        if not os.path.isfile(file_path):
            raise ValueError("File {} does not exists.".format(file_path))
        f = open(file_path, "r")
        for x in f:
            x = x.replace("\r\n", "\n").replace("\n", "")
            x = x.split(" ")
            if len(x) > 2:      # skip first line
                self.word2vec_embeddings[x[0]] = np.array(x[1:], dtype=float)

    def filter_model_with_polimorf(self):
        vocabulary = self._get_vocabulary_from_polimorf()
        new_word2vec_dictionary = dict()
        print("Number of words in Word2Vec: {}".format(len(self.word2vec_embeddings.keys())))
        for word, embedding in tqdm(self.word2vec_embeddings.items()):
            if word in vocabulary:
                new_word2vec_dictionary[word] = embedding
        return new_word2vec_dictionary

    def _get_vocabulary_from_polimorf(self):
        vocabulary = list()
        with open(os.path.join(DATA_PATH, "PoliMorf-0.6.7.tab"), 'r') as file:
            lines = file.readlines()
            for line in lines:
                words = line.split("\t")
                vocabulary.append(words[0])
        return vocabulary

    def get(self, word):
        """
        Return embedding for a word.
        :param word:
        :return: numpy.ndarray with word embedding
        """
        try:
            ret_val = self.model.get_vector(word)
        except:
            raise ValueError("Word {} is not avalaible in model.".format(word))
        return ret_val

    def semantic_relatedness(self, word1, word2, dist_type='cosine'):
        v1 = self.get(word1)
        v2 = self.get(word2)
        dist_fun = self.__get_distance_function(dist_type)
        return dist_fun(v1, v2)

    def synonyms(self, word, top=10, dist_type='cosine'):
        dist_fun = self.__get_distance_function(dist_type)
        distances = {}
        for neighbour in self.model.vocab:
            distances[neighbour] = dist_fun(self.get(word), self.get(neighbour))
        sorted_dist = sorted(distances.items(), key=lambda kv: kv[1])
        closest = [sd[0] for sd in sorted_dist[:top]]
        return closest

    def __get_distance_function(self, dist_type='cosine'):
        if dist_type == 'euclidean':
            return distance.euclidean
        elif dist_type == 'manhattan':
            return distance.cityblock
        elif dist_type == "cosine":
            return distance.cosine

    def save_model(self, file_name):
        print(type(self.model.vocab))
        print(self.model.vectors.shape)
        file_path = os.path.join(DATA_PATH, file_name + ".bin")
        self.model.save_word2vec_format(file_path, binary=True)

