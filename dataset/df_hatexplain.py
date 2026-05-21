import json
import requests
import pandas as pd

url = "https://raw.githubusercontent.com/punyajoy/HateXplain/master/Data/dataset.json"
data = json.loads(requests.get(url).text)

rows_hatexplain = []

for post_id, content in data.items():
    text = " ".join(content["post_tokens"])
  
    votes = [ann["label"] for ann in content["annotators"]]
    label = max(set(votes), key=votes.count)

    rows_hatexplain.append({
        "text": text,
        "label": label,
        "source": "hatexplain"
    })

df_hx = pd.DataFrame(rows_hatexplain)

df_hx["text"] = df_hx["text"].apply(clean_text)
df_hx = df_hx[df_hx["text"].apply(is_valid)]

print(df_hx.head())
