import copy
import json
from cmath import isnan

def get_data(filename):
    with open(filename) as f:
        data = json.load(f)
    return data

def get_triggers(data):
    all_triggers = []
    for conv in data:
        all_triggers.extend(conv["triggers"])

    for i in range(0, len(all_triggers)):
        if (isnan(all_triggers[i])):
            all_triggers[i] = 0.0
    return all_triggers

def rule_based(data):
    all_results = []

    # data = data[:10]
    for index in range (0, len(data)):
        item = copy.deepcopy(data[index])
        conv_len = len(item['speakers'])
        speakers_set = set(item['speakers'])
        result = [0.0]*conv_len
        speakers_dict = {}
        for name in speakers_set:
            speakers_dict[name] = []
        for i in range(0, conv_len):
            obj = {
                "index":i,
                "emotion": item["emotions"][i]
            }
            speakers_dict[item["speakers"][i]].append(obj)
        for name in speakers_set:
            evolution = copy.deepcopy(speakers_dict[name])

            for i in range(0,len(evolution)-1):
                if(evolution[i]["emotion"]!=evolution[i+1]["emotion"]):
                    toggled_speakers = []
                    first_index = evolution[i]["index"]
                    second_index = evolution[i+1]["index"]

                    if (second_index - 1 >= conv_len / 2):
                        result[second_index - 1] = 1.0
                    toggled_speakers.append(item["speakers"][second_index - 1])

        all_results.extend(result)

    return all_results


test_data = get_data('MELD_test_efr.json')

y_pred_test = rule_based(test_data)

with open("rule-based-test-results.txt", "w") as f:
    for item in y_pred_test:
        f.write(str(item+0.0) + "\n")