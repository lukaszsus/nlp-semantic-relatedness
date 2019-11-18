"""
Script created for testing and learning plwn library.
"""

import plwn


def try_plwn():
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
    print()
    print("Relation types:")
    realtion_set = set(relation_types)
    unique_relation_types = (list(realtion_set))
    for rel_type in unique_relation_types:
        print(rel_type)
    print(len(unique_relation_types))


def try_to_find_rel():
    wn = plwn.load_default()        # load wordnet
    # lexical_relation_edges = wn.lexical_relation_edges()        # load all lexical relations
    lexical_relation_edges = wn.lexical_relation_edges()
    rel_examples = dict()

    for l_rel_edge in lexical_relation_edges:
        src = l_rel_edge.source
        target = l_rel_edge.target
        rel = l_rel_edge.relation
        rel_name = rel.name
        if rel_name not in rel_examples.keys():
            rel_examples[rel_name] = (src.lemma, target.lemma)
    print()
    print("Relation examples:")
    for key, val in rel_examples.items():
        print("{}: {}".format(key, val))


if __name__ == '__main__':
    try_to_find_rel()