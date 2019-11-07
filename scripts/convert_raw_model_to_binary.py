import time
from models.ipi_pan_model import IpiPanModel
from data_loader import load_simlex_dataset

def convert_raw_to_binary(filter = False):
    """
    Converts Word2Vec model saved in txt file to binary and saves it as binary.
    :param filter: if model should be filtered with vocabulary from PoliMorf 0.6.7
    :return:
    """
    file_name = "wiki-forms-all-100-cbow-hs"
    start = time.time()
    model = IpiPanModel(file_name + ".txt")
    print("Loading took {}".format(time.time() - start))
    print(model.get("organizację"))
    if filter:
        start = time.time()
        model.filter_model_with_polimorf()
        print("Filtering model took {}".format(time.time() - start))
        file_name = file_name + "-filtered"

    start = time.time()
    model.save(file_name)
    print("Saving model took {}".format(time.time() - start))


if __name__ == '__main__':
    simlex_data = load_simlex_dataset()
    print(simlex_data)
    #convert_raw_to_binary(True)