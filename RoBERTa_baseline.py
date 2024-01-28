import numpy as np
import pandas as pd

import tensorflow as tf
from transformers import RobertaTokenizer, TFRobertaModel

import warnings
import numpy
import json
import os
import torch

from keras import backend as K

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2

warnings.filterwarnings("ignore")

try:
    # TPU detection. No parameters necessary if TPU_NAME environment variable is set (always set in Kaggle)
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    print('Running on TPU ', tpu.master())
except ValueError:
    # Default distribution strategy in Tensorflow. Works on CPU and single GPU.
    strategy = tf.distribute.get_strategy()

MODEL_NAME = 'roberta-base'
MAX_LEN = 100
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
EPOCHS = 3
ARTIFACTS_PATH = 'artifacts/'

print('Number of replicas:', strategy.num_replicas_in_sync)

if not os.path.exists(ARTIFACTS_PATH):
    os.makedirs(ARTIFACTS_PATH)

with open('dataset/task3_train_attributes.json') as f:
    X_train_json = json.load(f)

with open('dataset/task3_train_classes.json') as f:
    Y_train_json = json.load(f)

with open('dataset/task3_val_attributes.json') as f:
    X_val_json = json.load(f)

with open('dataset/task3_val_classes.json') as f:
    Y_val_json = json.load(f)

with open('dataset/task3_test_attributes.json') as f:
    X_test_json = json.load(f)

v = DictVectorizer(sparse=False)

#############################################################################3

X_train = v.fit_transform(X_train_json)

selector = SelectKBest(chi2, k=MAX_LEN)

X_train = selector.fit_transform(X_train, Y_train_json)
ct = len(X_train)
token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')

train_features = {
    'input_word_ids': X_train,
    'input_type_ids': token_type_ids
}

##########################################################################

X_test = v.transform(X_test_json)
# X_test = v.fit_transform(X_test_json)
ct = len(X_test)
token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
# support = SelectKBest(chi2, k=256).fit(X_test)

# v.restrict(support.get_support())
X_test = selector.transform(X_test)
# X_test = torch.tensor(X_test)

test_features = {
    'input_word_ids': X_test,
    'input_type_ids': token_type_ids
}

##########################################################################

X_val = v.transform(X_val_json)
ct = len(X_val)
token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')
# support = SelectKBest(chi2, k=256).fit(X_val, Y_val_json)
#
# v.restrict(support.get_support())
X_val = selector.transform(X_val)
# X_val = torch.tensor(X_val)

val_features = {
    'input_word_ids': X_val,
    'input_type_ids': token_type_ids
}

##########################################################################


def roberta_encode(texts, tokenizer):
    ct = len(texts)
    input_ids = np.ones((ct, MAX_LEN), dtype='int32')
    attention_mask = np.zeros((ct, MAX_LEN), dtype='int32')
    token_type_ids = np.zeros((ct, MAX_LEN), dtype='int32')  # Not used in text classification

    for k, text in enumerate(texts):
        # Tokenize
        tok_text = tokenizer.tokenize(text)

        # Truncate and convert tokens to numerical IDs
        enc_text = tokenizer.convert_tokens_to_ids(tok_text[:(MAX_LEN - 2)])

        input_length = len(enc_text) + 2
        input_length = input_length if input_length < MAX_LEN else MAX_LEN

        # Add tokens [CLS] and [SEP] at the beginning and the end
        input_ids[k, :input_length] = np.asarray([0] + enc_text + [2], dtype='int32')

        # Set to 1s in the attention input
        attention_mask[k, :input_length] = 1

    return {
        'input_word_ids': input_ids,
        'input_mask': attention_mask,
        'input_type_ids': token_type_ids
    }


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


def build_model(n_categories):
    with strategy.scope():
        input_word_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_word_ids')
        input_type_ids = tf.keras.Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_type_ids')

        # Import RoBERTa model from HuggingFace
        roberta_model = TFRobertaModel.from_pretrained(MODEL_NAME)
        x = roberta_model(input_word_ids, token_type_ids=input_type_ids)

        # Huggingface transformers have multiple outputs, embeddings are the first one,
        # so let's slice out the first position
        x = x[0]

        x = tf.keras.layers.Dropout(0.1)(x)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(MAX_LEN, activation='relu')(x)
        x = tf.keras.layers.Dense(n_categories, activation='softmax')(x)

        model = tf.keras.Model(inputs=[input_word_ids, input_type_ids], outputs=x)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=1e-5),
            loss='sparse_categorical_crossentropy',
            metrics=['acc', f1_m, precision_m, recall_m])

        return model


tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

# X_train = roberta_encode(X_train, tokenizer)
# X_val = roberta_encode(X_val, tokenizer)

Y_train = np.asarray(Y_train_json, dtype='int32')
Y_val = np.asarray(Y_val_json, dtype='int32')

with strategy.scope():
    model = build_model(2)
    model.summary()

with strategy.scope():
    print('Training...')
    history = model.fit(train_features,
                        Y_train,
                        # epochs=EPOCHS,
                        epochs=1,
                        batch_size=BATCH_SIZE,
                        verbose=1,
                        validation_data=(val_features, Y_val))

y_prob = model.predict(test_features)
y_pred_test = y_prob.argmax(axis=-1)

print(y_pred_test)

with open("output/output.txt", "w") as f:
    for x in y_pred_test:
        f.write(str(x+0.0) + "\n")

count = np.count_nonzero(y_pred_test)
print("count of 1 :", count)

print("Evaluating...")
loss, accuracy, f1_score, precision, recall = model.evaluate(val_features, Y_val, verbose=0)
print("Loss: %.2f%%" % (loss * 100))
print("Accuracy: %.2f%%" % (accuracy * 100))
print("F1: %.2f%%" % (f1_score * 100))
print("Precision: %.2f%%" % (precision * 100))
print("Recall: %.2f%%" % (recall * 100))
