from dicee import KGE

model = KGE("Experiments/2023-04-29 18_38_33.196057")

# x isa	entity AND NOT(x is spatial_concept)
entity_scores2 = []
atom1_scores = model.predict(head_entities=['cell_or_molecular_dysfunction'], relations=['process_of'])
print(len(atom1_scores))
atom1_scores = atom1_scores.squeeze()

atom2_scores = 1 - model.predict(head_entities=['congenital_abnormality'], relations=['part_of'])
atom2_scores = atom2_scores.squeeze()

assert len(atom1_scores) == len(model.entity_to_idx)
entity_scores = []
for ei, s1, s2 in zip(model.entity_to_idx.keys(), atom1_scores, atom2_scores):
    if s1 > s2:
        entity_scores.append((ei, float(s2)))
    else:
        entity_scores.append((ei, float(s1)))

entity_scores = sorted(entity_scores, key=lambda x: x[1], reverse=True)

print(entity_scores[:10])
