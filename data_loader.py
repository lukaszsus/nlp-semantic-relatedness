from settings import DATA_PATH


def load_text_file(filename):
    dataset = []
    f = open(f"{DATA_PATH}/{filename}", "r")
    for x in f:
        x = x.replace("\r\n", "\n").replace("\n", "")
        dataset.append(x)

    return " ".join(dataset)