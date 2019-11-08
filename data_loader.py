from settings import DATA_PATH
import pandas as pd
import os
import numpy as np


def load_text_file(filename):
    dataset = []
    f = open(f"{DATA_PATH}/{filename}", "r")
    for x in f:
        x = x.replace("\r\n", "\n").replace("\n", "")
        dataset.append(x)

    return " ".join(dataset)


def load_text_file_ad_array(filename):
    dataset = []
    f = open(f"{DATA_PATH}/{filename}", "r")
    for x in f:
        x = x.replace("\r\n", "\n").replace("\n", "").split('\t')[1:]
        x = [t.replace(" ", "") for t in x]
        dataset.append(x)

    return dataset


def load_simlex_dataset():
    file = load_text_file_ad_array("MSimLex999_Polish.txt")
    df = pd.DataFrame(data=file,
                      columns=['word1', 'word2', 'similarity', 'relatedness'])
    return df


def get_vocabulary_for_simlex():
    file = np.array(load_text_file_ad_array("MSimLex999_Polish.txt"))
    words = set(file[:, 0])
    words = words.union(set(file[:, 1]))
    return words


def get_vocabulary_from_polimorf():
    """
    PoliMorf available: http://zil.ipipan.waw.pl/PoliMorf?action=AttachFile&do=get&target=PoliMorf-0.6.7.tab.gz
    :return:
    """
    vocabulary = list()
    with open(os.path.join(DATA_PATH, "PoliMorf-0.6.7.tab"), 'r') as file:
        lines = file.readlines()
        for line in lines:
            words = line.split("\t")
            vocabulary.append(words[0])

    print(len(set(vocabulary)))
    print(len(vocabulary))
    return set(vocabulary)
