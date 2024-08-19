from dicee import KGE
# (1) Load a pretrained KGE model on KGs/Family
pre_trained_kge = KGE(path='Experiments/2023-05-10 18_49_27.078713')
# (2) Answer the following conjunctive query question: To whom a sibling of F9M167 is married to?
# (3) Decompose (2) into two query
# (3.1) Who is a sibling of F9M167? => {F9F141,F9M157}
# (3.2) To whom a results of (3.1) is married to ? {F9M142, F9F158}
# pretrained_model.predict_conjunctive_query(entity='<http://www.benchmark.org/family#F9M167>',
#                                           relations=['<http://www.benchmark.org/family#hasSibling>',
#                                                      '<http://www.benchmark.org/family#married>'], topk=1)

pre_trained_kge.triple_score(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],relation=["http://dbpedia.org/ontology/birthPlace"],tail_entity=["http://dbpedia.org/resource/Ulm"]) # tensor([0.9309])
pre_trained_kge.triple_score(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],relation=["http://dbpedia.org/ontology/birthPlace"],tail_entity=["http://dbpedia.org/resource/German_Empire"]) # tensor([0.9981])
pre_trained_kge.triple_score(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],relation=["http://dbpedia.org/ontology/birthPlace"],tail_entity=["http://dbpedia.org/resource/Kingdom_of_Württemberg"]) # tensor([0.9994])
pre_trained_kge.triple_score(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],relation=["http://dbpedia.org/ontology/birthPlace"],tail_entity=["http://dbpedia.org/resource/Germany"]) # tensor([0.9498])
pre_trained_kge.triple_score(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],relation=["http://dbpedia.org/ontology/birthPlace"],tail_entity=["http://dbpedia.org/resource/France"]) # very low
pre_trained_kge.triple_score(head_entity=["http://dbpedia.org/resource/Albert_Einstein"],relation=["http://dbpedia.org/ontology/birthPlace"],tail_entity=["http://dbpedia.org/resource/Italy"]) # very low