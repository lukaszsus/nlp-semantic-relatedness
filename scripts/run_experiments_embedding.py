import data_loader as data_loader
from models.ipi_pan_model import IpiPanModel
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd

results_columns = ['metric', 'similarity_func', 'similarity_cor',
                   'relatedness_cor', 'model_name', 'missed_words',
                   'unique_missed_words']

models = {
    "nkjp+wiki": "nkjp+wiki-forms-restricted-100-cbow-hs.txt-filtered.bin"
}

reverted_models = {value: key for key, value in models.items()}

metrics = {
    "scikit.pearsonr": lambda x, y: pearsonr(x, y)[0],
    "scikit.spearmanr": lambda x, y: spearmanr(x, y)[0]
}

reverted_metrics = {value: key for key, value in metrics.items()}

similarity_functions = {
    "manhattan": "manhattan",
    "cosine": "cosine",
    "euclidean": "euclidean"
}

reverted_similarity_functions = {value: key for key, value in
                                 similarity_functions.items()}


def get_values_to_check(model, test_dataset):
    values_to_check = []
    missed_words = []

    for i, row in test_dataset.iterrows():
        word1 = row[['word1']].values[0]
        word2 = row[['word2']].values[0]
        contains_word_1 = model.contains(word1)
        contains_word_2 = model.contains(word2)

        if contains_word_1 and contains_word_2:
            values_to_check.append(row.values)
        else:
            if not contains_word_1:
                missed_words.append(word1)
            elif not contains_word_2:
                missed_words.append(word2)

    print(missed_words)
    return values_to_check, len(missed_words), len(set(missed_words))


def test_model(model, values_to_check, metric, similarity_function):
    print("------")

    results = []

    for values in values_to_check:
        word1 = values[0]
        word2 = values[1]
        similarity = values[2]
        relatedness = values[3]

        counted = model.semantic_relatedness(word1, word2,
                                             dist_type=similarity_function)

        results.append([similarity, relatedness, counted])

    results = np.array(results).astype("float32").transpose()

    model_result = {
        "metric": metric,
        "similarity_func": reverted_similarity_functions[similarity_function],
        "similarity_cor": metrics[metric](results[0, :], results[2, :]),
        "relatedness_cor": metrics[metric](results[1, :], results[2, :]),
    }

    return model_result


def main():
    test_dataset = data_loader.load_simlex_dataset()
    print(test_dataset)

    results = pd.DataFrame(columns=results_columns)

    for model_name in models.keys():
        model = IpiPanModel(models[model_name])

        values_to_check, missed_words, unique_missed_words = get_values_to_check(
            model, test_dataset)

        for similarity_function in similarity_functions.keys():
            for metric in metrics:
                print(
                    f"Testing {model_name} : {similarity_function} : {metric}")

                model_results = test_model(model=model,
                                           values_to_check=values_to_check,
                                           metric=metric,
                                           similarity_function=similarity_function)

                model_results['model_name'] = model_name
                model_results['missed_words'] = missed_words
                model_results['unique_missed_words'] = unique_missed_words

                model_results_df = pd.DataFrame(
                    data=[model_results], columns=results_columns)
                results = pd.concat([results, model_results_df])

    print(results)


if __name__ == '__main__':
    main()
