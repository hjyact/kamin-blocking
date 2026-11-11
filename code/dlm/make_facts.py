# fictional birthplace facts for the masked-diffusion blocking test: half the people are stated twice within one document (dup),
# half once in each of two documents (split); same templates and token counts. Writes facts.json
import json, random, tiktoken
enc = tiktoken.get_encoding("gpt2"); random.seed(0)
cands = "Lisbon Oslo Cairo Lima Hanoi Dublin Quito Nairobi Seoul Prague Vienna Havana Manila Athens Warsaw Bogota Dakar Riga Tunis Kyiv Accra Muscat Beirut Tbilisi Yerevan Baku Doha Amman Sofia Zagreb Belgrade Minsk Vilnius Tallinn Helsinki Madrid Rome Paris Berlin London Tokyo Delhi Lagos Sydney Toronto Chicago Boston Dallas Denver Miami Seattle Phoenix Houston Austin Detroit Moscow Beijing Shanghai Mumbai Bangkok Jakarta Singapore Istanbul Tehran Baghdad Venice Florence Naples Milan Munich Hamburg Vancouver Montreal Atlanta Portland Orlando Memphis Nashville Oakland Cleveland Pittsburgh Baltimore Glasgow Edinburgh Dublin Manchester Liverpool Bristol Oxford Cambridge".split()
cities = [c for c in cands if len(enc.encode(" " + c)) == 1][:50]; assert len(cities) == 50, len(cities)
FIRST = "Alder Brisa Corvin Dalia Emrys Fenna Galen Hesper Ilsa Joren Kestrel Liora Maelis Nyra Orrin Pell Quill Rowan Sabine Tamsin".split()
LAST = "Ashgrove Blackmere Crowhurst Dunmore Elswick Farrow Greywell Halloran Iverson Jessop Kettering Larkspur Morrow Northcott Oakhurst Pemberton Quarry Ravensworth Stillwater Thorne".split()
people = random.sample([f"{a} {b}" for a in FIRST for b in LAST], 400); city = [random.randrange(50) for _ in people]
cond = ["dup" if i % 2 == 0 else "split" for i in range(len(people))]
T1 = "{n} was born in {c}."; T2 = "As a child, {n} lived in {c}."
docs = []
for i, n in enumerate(people):
    c = cities[city[i]]
    if cond[i] == "dup": docs.append([enc.encode(" " + T1.format(n=n, c=c) + " " + T2.format(n=n, c=c))])
    else: docs.append([enc.encode(" " + T1.format(n=n, c=c)), enc.encode(" " + T2.format(n=n, c=c))])
json.dump(dict(cities=cities, city_ids=[enc.encode(" " + c)[0] for c in cities], people=people, city=city, cond=cond, docs=docs,
               prompt=[enc.encode(" " + n + " was born in") for n in people], dot=enc.encode(".")), open("facts.json", "w"))
print("ok", len(people), sum(len(d) for d in docs), "docs")
