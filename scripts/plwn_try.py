"""
Script created for testing and learning plwn library.
"""

import plwn

if __name__ == '__main__':
    wn = plwn.load_default()        # load wordnet
    lexical_relation_edges = wn.lexical_relation_edges()        # load all lexical relations
    relations = list()
    relation_types = list()
    for l_rel_edge in lexical_relation_edges:
        src = l_rel_edge.source
        src_lemma = src.lemma
        target = l_rel_edge.target
        target_lemma = target.lemma
        rel = l_rel_edge.relation
        print(rel)
        rel_name = rel.name
        # print(src_lemma + ";", target_lemma + ";", rel_name + ";", rel.aliases)
        relations.append(rel)
        relation_types.append(rel_name)
    # get_unique
    print()
    print("Relation types:")
    realtion_set = set(relation_types)
    unique_relation_types = (list(realtion_set))
    for rel_type in unique_relation_types:
        print(rel_type)
    print(len(unique_relation_types))

    print("\n\n==========================")
    for rel_type in relations:
        print(rel_type)



    # word = "piec"
    # lch_threshold = 2.26
    # wn = plwn.load_default()

    # def get_all_synsets(word, pos=None):
    #     for ss in wn.synsets(word):
    #         for lexical_unit in ss.lexical_units:
    #             yield (lexical_unit.lemma, ss.name())
    #
    #
    # def get_all_hyponyms(word, pos=None):
    #     for ss in wn.synsets(word, pos=pos):
    #         for hyp in ss.hyponyms():
    #             for lemma in hyp.lemmas():
    #                 yield (lemma, hyp.name())
    #
    #
    # def get_all_similar_tos(word, pos=None):
    #     for ss in wn.synsets(word):
    #         for sim in ss.similar_tos():
    #             for lemma in sim.lemma_names():
    #                 yield (lemma, sim.name())
    #
    #
    # def get_all_antonyms(word, pos=None):
    #     for ss in wn.synsets(word, pos=None):
    #         for sslema in ss.lemmas():
    #             for antlemma in sslema.antonyms():
    #                 yield (antlemma.name(), antlemma.synset().name())
    #
    #
    # def get_all_also_sees(word, pos=None):
    #     for ss in wn.synsets(word):
    #         for also in ss.also_sees():
    #             for lemma in also.lemma_names():
    #                 yield (lemma, also.name())
    #
    #
    # def get_all_synonyms(word, pos=None):
    #     for x in get_all_synsets(word, pos):
    #         yield (x[0], x[1], 'ss')
    #     for x in get_all_hyponyms(word, pos):
    #         yield (x[0], x[1], 'hyp')
    #     for x in get_all_similar_tos(word, pos):
    #         yield (x[0], x[1], 'sim')
    #     for x in get_all_antonyms(word, pos):
    #         yield (x[0], x[1], 'ant')
    #     for x in get_all_also_sees(word, pos):
    #         yield (x[0], x[1], 'also')
    #
    # for x in get_all_synonyms('love'):
    #     print(x)

    # wn = plwn.load_default()        # ładowanie zrzutu bazy danych (jest już tam cały wordnet)
    # wn.synsets()                    # zwraca wszystkie synsety, można po nich iterować
    # for s in wn.synsets():
    #     s.id
    #     s.lexical_units     # synonimy (w ramach jednego synsetu)
    #
    # for l in wn.lexical_units():
    #     l.lemma, l.pos, l.variant
    #     l.definition            #definicja (jak ze słownika, encyklopedii)
    #     l.sense_example             # przykład użycia (kontekst)
    #
    # # Mamy dostęp do dwóch poziomów relacji:
    # wn.synsets_relations()          # wszystkie relacje
    # wn.synsets_relations()          # wszystkie relacje
    # s = wn.sysnets()[0]
    # s.relations()           # wszystkie relacje synsetów
    # pairs = s.related_pairs()       # synset i relacja
    #
    # l = wn.lexical_units()[0]
    # l.relations()           # relacje jednostki
    # l.related_pairs()       # pary relacji