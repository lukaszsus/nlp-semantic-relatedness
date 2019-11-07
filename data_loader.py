from settings import DATA_PATH
import pandas as pd


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
        dataset.append(x)

    return dataset


def load_simlex_dataset():
    file = load_text_file_ad_array("MSimLex999_Polish.txt")
    df = pd.DataFrame(data=file, columns=['word1', 'word2', 'similarity', 'relatedness'])
    return df