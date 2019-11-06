"""
Since 6.11.2019 we do not use gensim so this is deprecated.
It is not deleted because of possibility of returning to this idea.
"""


import time
from models.ipi_pan_model_gensim import IpiPanModelGensim

def convert_raw_to_binary(filter = False):
    start = time.time()
    model = IpiPanModelGensim("wiki-forms-all-100-cbow-hs.txt")
    print("Loading took {}".format(time.time() - start))

    if filter:
        start = time.time()
        model.filter_model_with_polimorf()
        print("Filtering model took {}".format(time.time() - start))

    start = time.time()
    model.save_model("wiki-forms-all-100-cbow-hs-filtered")
    print("Saving model took {}".format(time.time() - start))

if __name__ == '__main__':
    convert_raw_to_binary()