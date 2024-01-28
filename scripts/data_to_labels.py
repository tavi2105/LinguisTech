import json
from cmath import isnan

with open('../dataset/MELD_train_efr.json') as f:
    task3_train_data = json.load(f)

with open('../dataset/MELD_val_efr.json') as f:
    task3_val_data = json.load(f)

with open('../dataset/MELD_test_efr.json') as f:
    task3_test_data = json.load(f)

train = []
test = []
yesno = ['no', 'yes']

for data in task3_train_data:
    labels = [yesno[int(i)] for i in data["triggers"]]
    train.append({"tokens": data["emotions"], "labels": labels})

for data in task3_val_data:
    labels = [yesno[int(i)] for i in data["triggers"]]
    train.append({"tokens": data["emotions"], "labels": labels})

for data in task3_test_data:
    test.append(" ".join(data["emotions"]))


json_object = json.dumps(train, indent=4)
with open("../dataset/train_labels.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object)

json_object = json.dumps(test, indent=4)
with open("../dataset/test_labels.json", "w") as outfile:
    outfile.truncate()
    outfile.write(json_object)
