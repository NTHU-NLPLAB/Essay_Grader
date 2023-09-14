import pickle
import requests
import pandas as pd
import json
from sklearn.ensemble import RandomForestRegressor
import collections
from flask import Flask, request, jsonify

app = Flask(__name__)

with open('features.json', 'r') as f:
    features_label = json.load(f)
reg = pickle.load(open("random_forest.pickle", 'rb'))

def flatten(dictionary, parent_key=False, separator='_'):
    """
    Turn a nested dictionary into a flattened dictionary
    :param dictionary: The dictionary to flatten
    :param parent_key: The string to prepend to dictionary's keys
    :param separator: The string used to separate flattened keys
    :return: A flattened dictionary
    """

    items = []
    for key, value in dictionary.items():
        new_key = str(parent_key) + separator + key if parent_key else key
        if isinstance(value, collections.abc.MutableMapping):
            items.extend(flatten(value, new_key, separator).items())
        elif isinstance(value, list):
            for k, v in enumerate(value):
                items.extend(flatten({str(k): v}, new_key).items())
        else:
            items.append((new_key, value))
    return dict(items)


def features_extraction(text):
    """
    Takes a dict {"text": string}
    and produce feature vector based on pineapple-api 
    """
    score_conversion = {'A1': 1, 'A2': 2, 'B1': 3, "B2": 4, 'C1': 5, 'C2':6}
    api_url = "https://whisky.nlplab.cc/api/aes/feedback/"
    res = requests.post(api_url, json=text)
    print("pineapple Automated Essay Analysis: ", res.json())
    data = flatten(res.json())
    api_url = "https://whisky.nlplab.cc/api/aes/"
    res = requests.post(api_url, json=text)
    print("pineapple Automated Essay Scoring: ", res.json())
    data['essay_level'] = score_conversion[res.json()['score']]
    return data

@app.route("/grader/", methods = {'POST'})
def essay_grader():
    """
    Please post with the following json argument
    {"text": "This is a sample."}
    """
    essay = request.get_json()
    features = features_extraction(essay)
    vector = pd.DataFrame.from_dict([features])
    vector = vector.reindex(features_label, axis=1, fill_value=0)
    grade = reg.predict(vector)[0]
    grade = round(grade)
    
    return jsonify({"grade":grade}), 201



if __name__ == "__main__":
    app.run(debug=True)
    
