import json
from cmath import isnan

with open('../dataset/MELD_train_efr.json') as f:
    task3_train_data = json.load(f)

X_train = []
Y_train = []

for index in range(0, len(task3_train_data)):
    item = task3_train_data[index]
    for i in range(int(len(item["speakers"])/2), len(item["speakers"])):
        entry = {}
        entry["emotion"] = item["emotions"][i]
        entry["speaker"] = item["speakers"][i]

        X_train.append(entry)
        if not isnan(item["triggers"][i]):
            Y_train.append(item["triggers"][i])
        else:
            Y_train.append(0.0)

json_object_X = json.dumps(X_train, indent=4)
with open("../dataset/task3_train_attributes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_X)

json_object_Y = json.dumps(Y_train, indent=4)
with open("../dataset/task3_train_classes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_Y)

# ##########################################################################################################################

with open('../dataset/MELD_val_efr.json') as f:
    task3_val_data = json.load(f)

X_test = []
Y_test = []

for index in range(0, len(task3_val_data)):
    item = task3_val_data[index]
    for i in range(0, len(item["speakers"])):
        entry = {}
        entry["emotion"] = item["emotions"][i]
        entry["speaker"] = item["speakers"][i]

        X_test.append(entry)
        if (not isnan(item["triggers"][i])):
            Y_test.append(item["triggers"][i])
        else:
            Y_test.append(0.0)

json_object_X = json.dumps(X_test, indent=4)
with open("../dataset/task3_val_attributes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_X)

json_object_Y = json.dumps(Y_test, indent=4)
with open("../dataset/task3_val_classes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_Y)

# ##########################################################################################################################

with open('../dataset/MELD_test_efr.json') as f:
    task3_test_data = json.load(f)

X_test = []
Y_test = []

for index in range(0, len(task3_test_data)):
    item = task3_test_data[index]
    for i in range(0, len(item["speakers"])):
        entry = {}
        entry["emotion"] = item["emotions"][i]
        entry["speaker"] = item["speakers"][i]

        X_test.append(entry)

json_object_X = json.dumps(X_test, indent=4)
with open("../dataset/task3_test_attributes.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object_X)
