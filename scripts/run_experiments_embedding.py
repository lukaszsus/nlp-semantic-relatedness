import data_loader as data_loader
from models.ipi_pan_model import IpiPanModel
from models.fast_text_gensim import FastTextGensim
from settings import OUTPUT_PATH
import numpy as np
from scipy.stats import spearmanr, pearsonr
import pandas as pd

results_columns = ['metric', 'similarity_func', 'similarity_cor',
                   'relatedness_cor', 'model_name', 'missed_words_array', 'missed_words',
                   'unique_missed_words']

models = {
    "w2v-nkjp+wiki-forms-restricted-100-cbow-hs.txt-filtered": "nkjp+wiki-forms-restricted-100-cbow-hs.txt-filtered.bin",
    "fasttext-kgr10.plain.lemma.lower.cbow.dim100.neg10.bin-filtered": "kgr10.plain.lemma.lower.cbow.dim100.neg10.bin-filtered.bin"
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

    return values_to_check, set(missed_words), len(missed_words), len(set(missed_words))


def test_model(model, values_to_check, distance_function):
    print("------")

    results = []
    results_to_save = []

    for values in values_to_check:
        word1 = values[0]
        word2 = values[1]
        similarity = values[2]
        relatedness = values[3]

        counted = 1.0 / model.semantic_relatedness(word1, word2,
                                             dist_type=distance_function)

        results.append([similarity, relatedness, counted])
        results_to_save.append([word1, word2, similarity, relatedness, counted])

    results = np.array(results).astype("float32").transpose()
    results_to_save = pd.DataFrame(data=results_to_save, columns=["word1", "word2", 'similarity', "relatedness", "counted_similarity"])

    return results, results_to_save


def main():
    test_dataset = data_loader.load_simlex_dataset()
    print(test_dataset)

    results = pd.DataFrame(columns=results_columns)

    for model_name in models.keys():
        if model_name.startswith("fasttext"):
            model = FastTextGensim(models[model_name])
        elif model_name.startswith("w2v"):
            model = IpiPanModel(models[model_name])
        else:
            continue

        values_to_check, missed_words_array, missed_words, unique_missed_words = get_values_to_check(
            model, test_dataset)

        for similarity_function in similarity_functions.keys():
            run_results, run_results_pd = test_model(model=model,
                                                    values_to_check=values_to_check,
                                                    distance_function=similarity_function)

            run_results_pd.to_csv(f'{OUTPUT_PATH}/counted_results/{model_name}-{similarity_function}')

            for metric in metrics:
                model_result = {
                    "metric": metric,
                    "similarity_func": reverted_similarity_functions[similarity_function],
                    "similarity_cor": metrics[metric](run_results[0, :], run_results[2, :]),
                    "relatedness_cor": metrics[metric](run_results[1, :], run_results[2, :]),
                    "model_name": model_name,
                    "missed_words_array": missed_words_array,
                    "missed_words": missed_words,
                    "unique_missed_words": unique_missed_words,
                }

                model_results_df = pd.DataFrame(
                    data=[model_result], columns=results_columns)
                results = pd.concat([results, model_results_df])

    results = results.reset_index()
    results = results.drop(columns=["index"])
    print(results)
    results.to_csv(f"{OUTPUT_PATH}/embeddings_results.csv")


if __name__ == '__main__':
    main()
