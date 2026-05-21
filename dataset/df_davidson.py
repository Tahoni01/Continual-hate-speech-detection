import json
import requests
import pandas as pd
import re
from datasets import load_dataset

def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", "", text)   # URL
    text = re.sub(r"@\w+", "", text)                      # mentions
    text = re.sub(r"#", "", text)                         # hashtag symbol
    text = re.sub(r"[^a-z0-9\s]", "", text)               # special chars
    text = re.sub(r"\s+", " ", text).strip()              # spaces

    return text

def is_valid(text):
    return isinstance(text, str) and len(text.split()) >= 3

def getdf_hatexplain():
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
  
  # CLEAN HATEXPLAIN
  df_hx["text"] = df_hx["text"].apply(clean_text)
  df_hx = df_hx[df_hx["text"].apply(is_valid)]
  
  print(df_hx.head())

  return df_hx


def getdf_davidson():
  davidson = load_dataset("tdavidson/hate_speech_offensive")
  
  df_davidson = davidson["train"].to_pandas()
  
  label_map = {
      0: "hatespeech",
      1: "offensive",
      2: "normal"
  }
  
  df_davidson = df_davidson.rename(columns={"tweet": "text"})
  df_davidson["label"] = df_davidson["class"].map(label_map)
  
  df_davidson = df_davidson[["text", "label"]]
  df_davidson["source"] = "davidson"
  
  df_davidson["text"] = df_davidson["text"].apply(clean_text)
  df_dv = df_davidson[df_davidson["text"].apply(is_valid)]
  
  print(df_dv.head())

  return df_dv

def get_plot(df_dv,df_hx):
  summary = pd.DataFrame({
    "dataset": ["hatexplain", "davidson"],
    "samples": [len(df_hx), len(df_dv)]
  })
  
  print(summary)
