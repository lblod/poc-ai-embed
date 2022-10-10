import requests
from sklearn.metrics.pairwise import cosine_similarity
import json

url ="http://192.168.6.137:9999/predictions/EmbedBert/1.0"
req = requests.post(
    url,
    json={"texts":[
        {"text_id": 123, "text": "Een fiets is een voertuig"},
        {"text_id": 124, "text": "Een auto is een vervoersmiddel, automobiel, autosnelweg, autostrade, vervoer, vervoersmiddel"},
        {"text_id": 125, "text": "Een zwembad is een plaats waar mensen nat worden :eyes: "},
    ]}
)


print(req.text)

res = req.json()["texts"]
first = cosine_similarity([res[0]["embedding"]], [res[1]["embedding"]])
second = cosine_similarity([res[0]["embedding"]], [res[2]["embedding"]])
third = cosine_similarity([res[1]["embedding"]], [res[2]["embedding"]])

print("0-1: ",first,"| 0-2: ", second,"| 1-2: ", third)
