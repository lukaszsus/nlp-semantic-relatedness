import time
from user_settings import DATA_PATH
from models.ipi_pan_model import IpiPanModel
import data_loader as data_loader

import os


def convert_raw_to_binary(file_name, filter = False):
    """
    Converts Word2Vec model saved in txt file to binary and saves it as binary.
    :param filter: if model should be filtered with vocabulary from PoliMorf 0.6.7
    :return:
    """
    start = time.time()
    model = IpiPanModel(file_name)
    print("Loading took {}".format(time.time() - start))

    base_name, ext = os.path.splitext(file_name)

    if filter:
        start = time.time()
        polimorf_vocabulary = data_loader.get_vocabulary_from_polimorf()
        simplex_vocabulary = data_loader.get_vocabulary_for_simlex()
        print(simplex_vocabulary)
        vocabulary = polimorf_vocabulary.union(simplex_vocabulary)
        model.filter_model_with_polimorf(vocabulary)
        print("Filtering model took {}".format(time.time() - start))
        base_name = base_name + "-filtered"

    start = time.time()
    model.save(base_name)
    print("Saving model took {}".format(time.time() - start))
    del model


if __name__ == '__main__':
    # convert_raw_to_binary("nkjp+wiki-forms-restricted-100-cbow-hs.txt", True)
    # convert_raw_to_binary("nkjp+wiki-forms-restricted-100-skipg-hs.txt", True)
    convert_raw_to_binary("nkjp+wiki-forms-restricted-300-cbow-hs.txt", True)
    # convert_raw_to_binary("nkjp+wiki-forms-restricted-300-skipg-hs.txt", True)
