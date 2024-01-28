import torch
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from keras import backend as K
import copy
import json
from cmath import isnan
import keras_nlp
import keras
import numpy as np

import copy
import json
from cmath import isnan


def preprocess(data):
    X_data = []
    Y_data = []
    for index in range(0, len(data)):
        item = copy.deepcopy(data[index])
        for i in range(0, len(item["speakers"])):
            entry = {}
            entry["entry_index"] = i
            entry["entry"] = str(i) + " - " + item["speakers"][i] + " - " + item["utterances"][i] + " - " + \
                             item["emotions"][i]
            conv = ""
            for j in range(0, len(item["speakers"])):
                line = item["speakers"][j] + ": " + item["utterances"][j] + " - " + \
                       item["emotions"][j]
                conv = conv + line + "\n"
            entry["context"] = conv
            X_data.append(entry)

            if "triggers" in item:
                if (not isnan(item["triggers"][i])):
                    Y_data.append(item["triggers"][i])
                else:
                    Y_data.append(0.0)
    return X_data, Y_data


with open('dataset/MELD_train_efr.json') as f:
    task3_train_data = json.load(f)

with open('dataset/MELD_val_efr.json') as f:
    task3_val_data = json.load(f)

with open('dataset/MELD_test_efr.json') as f:
    task3_test_data = json.load(f)

X_train_data, Y_train = preprocess(task3_train_data)
X_val_data, Y_val = preprocess(task3_val_data)
X_test_data, Y_test = preprocess(task3_test_data)

v = DictVectorizer(sparse=False)

########################################################################

X_train = v.fit_transform(X_train_data)
support = SelectKBest(chi2, k=100).fit(X_train, Y_train)

v.restrict(support.get_support())
X_train = v.transform(X_train_data)
X_train = torch.tensor(X_train)

padding_masking = X_train > 0

padding_masking = padding_masking.int()

print(X_train.shape)
print(len(Y_train))

train_features = {
    "token_ids": X_train,
    "padding_mask": padding_masking
}

print(X_train)
print(train_features)

##########################################################################

X_test = v.transform(X_test_data)
X_test = torch.tensor(X_test)

padding_masking = X_test > 0

padding_masking = padding_masking.int()

test_features = {
    "token_ids": X_test,
    "padding_mask": padding_masking
}

############################################################################

X_val = v.fit_transform(X_val_data)
support = SelectKBest(chi2, k=100).fit(X_val, Y_val)

v.restrict(support.get_support())
X_val = v.transform(X_val_data)
X_val = torch.tensor(X_val)

padding_masking = X_val > 0

padding_masking = padding_masking.int()

val_features = {
    "token_ids": X_val,
    "padding_mask": padding_masking
}

#############################################################################

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_m(y_true, y_pred):
    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


classifier = keras_nlp.models.DistilBertClassifier.from_preset(
    "distil_bert_base_en",
    preprocessor=None,
    num_classes=2,
)

classifier.compile(
    loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(5e-5),
    jit_compile=True,
    metrics=['accuracy', f1_m, precision_m, recall_m]
)

classifier.fit(x=train_features, y=Y_train)

loss, accuracy, f1_score, precision, recall = classifier.evaluate(val_features, Y_val, verbose=False)
print("Testing Accuracy: {:.4f}".format(accuracy))
print("Testing F1: {:.4f}".format(f1_score))
print("Testing Precision: {:.4f}".format(precision))
print("Testing Recall: {:.4f}".format(recall))


y_prob = classifier.predict(test_features)
print(y_prob)
y_pred_test = y_prob.argmax(axis=-1)

with open("output/test-results.txt", "w") as f:
    for item in y_pred_test:
        f.write(str(item+0.0) + "\n")
