# packing test: four conditions of 100 fictional people each, same two sentences (T1 then T2) and the same number of mentions:
# dup (both in one document), split (one document each, separate sequences), pack (both documents in the same training sequence,
# T1 document in the first half, T2 document in the second), packm (as pack, but the T2 document cannot attend to the T1 document).
import json, random, tiktoken
exec(open("../dlm/make_facts.py").read().split("cond = ")[0])
cond = [["dup", "split", "pack", "packm"][i % 4] for i in range(len(people))]
T1 = "{n} was born in {c}."; T2 = "As a child, {n} lived in {c}."
docs, units = [], []
for i, n in enumerate(people):
    c = cities[city[i]]; a = enc.encode(" " + T1.format(n=n, c=c)); b = enc.encode(" " + T2.format(n=n, c=c))
    if cond[i] == "dup": docs.append([a + b]); units.append(dict(docs=[a + b], mask=False))
    elif cond[i] == "split": docs.append([a, b]); units += [dict(docs=[a], mask=False), dict(docs=[b], mask=False)]
    else: docs.append([a, b]); units.append(dict(docs=[a, b], mask=cond[i] == "packm"))
json.dump(dict(cities=cities, city_ids=[enc.encode(" " + c)[0] for c in cities], people=people, city=city, cond=cond, docs=docs, units=units,
               prompt=[enc.encode(" " + n + " was born in") for n in people], dot=enc.encode(".")), open("../dlm/facts_pack.json", "w"))
print("ok", len(people), len(units), "units")
