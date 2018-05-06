import reader
import numpy as np
from collections import Counter


class LabelEncoder:
    """
    Используется для сокращения представления меток (в работе оценок)
    """

    def __init__(self):
        self.data = {}
        self.index = 0

    def get(self, label):
        if label in self.data:
            return self.data[label]
        else:
            self.data[label] = self.index
            self.index += 1
            return self.data[label]


def docs_from_dataset(folder_path, file_name, column_types, used_columns, sent2features):
    dataset = reader.DataReader(folder_path, fileids=file_name, columntypes=column_types)
    y = [el[1] for el in dataset.get_ne()]
    x = dataset.get_tags(tags=used_columns)
    x_sent_base, y_sent = [], []
    index = 0
    for sent in dataset.sents():
        length = len(sent)
        if length == 0:
            continue
        x_sent_base.append(x[index:index + length])
        y_sent.append(y[index:index + length])
        index += length

    x_sent = [sent2features(s) for s in x_sent_base]

    x_docs, y_docs = [], []
    index = 0
    for doc in dataset.docs():
        length = len(doc)
        if length == 0:
            continue
        x_docs.append(x_sent[index:index + length])
        y_docs.append(y_sent[index:index + length])
        index += length
    return x_docs, y_docs


def docs_from_dataset_tokens(dataset, tags=['words', 'pos', 'chunk']):
    y = [el[1] for el in dataset.get_ne()]
    x = dataset.get_tags(tags=tags)

    x_sent, y_sent = [], []
    index = 0
    for sent in dataset.sents():
        length = len(sent)
        if length == 0:
            continue
        x_sent.append(x[index:index + length])
        y_sent.append(y[index:index + length])
        index += length
    x_docs, y_docs = [], []
    index = 0
    for doc in dataset.docs():
        length = len(doc)
        if length == 0:
            continue
        x_docs.append(x_sent[index:index + length])
        y_docs.append(y_sent[index:index + length])
        index += length
    return x_docs, y_docs


def xdocs_from_x_dataset(x, dataset):
    x_sent = []
    index = 0
    for sent in dataset.sents():
        length = len(sent)
        if length == 0:
            continue
        x_sent.append(x[index:index + length])
        index += length
    x_docs = []
    index = 0
    for doc in dataset.docs():
        length = len(doc)
        if length == 0:
            continue
        x_docs.append(x_sent[index:index + length])
        index += length
    return x_docs


def add_token_features(x_train_docs, y_train_docs_predicted):
    tokens_corp = dict()

    for x_doc, y_doc in zip(x_train_docs, y_train_docs_predicted):
        tokens_doc = dict()
        for x_sent, y_sent in zip(x_doc, y_doc):
            for x_token, y_token in zip(x_sent, y_sent):
                token = x_token['word.lower()']
                # prefixes
                if y_token != 'O':
                    y_token = y_token[2:]
                if token in tokens_corp:
                    tokens_corp[token].append(y_token)
                else:
                    tokens_corp[token] = [y_token]
                if token in tokens_doc:
                    tokens_doc[token].append(y_token)
                else:
                    tokens_doc[token] = [y_token]

        for key in tokens_doc.keys():
            tokens_doc[key] = Counter(tokens_doc[key])
        for x_sent in x_doc:
            for x_token in x_sent:
                x_token['doc_token_maj'] = tokens_doc[x_token['word.lower()']].most_common(1)[0][0]

    for key in tokens_corp.keys():
        tokens_corp[key] = Counter(tokens_corp[key])

    for x_doc in x_train_docs:
        for x_sent in x_doc:
            for x_token in x_sent:
                x_token['corp_token_maj'] = tokens_corp[x_token['word.lower()']].most_common(1)[0][0]


def add_entity_features(x_train_docs, y_train_docs_predicted):
    entities_corp = dict()
    current_entity = ''

    for x_doc, y_doc in zip(x_train_docs, y_train_docs_predicted):
        entities_doc = dict()
        for x_sent, y_sent in zip(x_doc, y_doc):
            current_indexes = []
            current_entity = ''
            for x_token, y_token in zip(x_sent, y_sent):
                token = x_token['word.lower()']
                if y_token[0] == 'S':
                    current_entity = token
                    if current_entity in entities_doc:
                        entities_doc[current_entity].append(y_token[2:])
                    else:
                        entities_doc[current_entity] = [y_token[2:]]
                    if current_entity in entities_corp:
                        entities_corp[current_entity].append(y_token[2:])
                    else:
                        entities_corp[current_entity] = [y_token[2:]]
                if y_token[0] == 'B':
                    current_entity = token
                if y_token[0] == 'I':
                    current_entity += ' ' + token
                if y_token[0] == 'E':
                    current_entity += ' ' + token
                    if current_entity in entities_doc:
                        entities_doc[current_entity].append(y_token[2:])
                    else:
                        entities_doc[current_entity] = [y_token[2:]]
                    if current_entity in entities_corp:
                        entities_corp[current_entity].append(y_token[2:])
                    else:
                        entities_corp[current_entity] = [y_token[2:]]
                    current_entity = ''
                if y_token == 'O':
                    current_entity = ''
        for key in entities_doc.keys():
            entities_doc[key] = Counter(entities_doc[key])
        for x_sent, y_sent in zip(x_doc, y_doc):
            current_indexes = []
            current_entity = ''
            for i in range(len(x_sent)):
                if y_sent[i][0] == 'S':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                    for j in current_indexes:
                        x_sent[j]['doc_entity_maj'] = entities_doc[current_entity].most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []
                if y_sent[i][0] == 'B':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                if y_sent[i][0] == 'I':
                    current_entity = current_entity + ' ' + x_sent[i]['word.lower()']
                    current_indexes.append(i)
                if y_sent[i][0] == 'E':
                    current_entity = current_entity + ' ' + x_sent[i]['word.lower()']
                    current_indexes.append(i)
                    if current_entity in entities_doc:
                        for j in current_indexes:
                            x_sent[j]['doc_entity_maj'] = entities_doc[current_entity].most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []
                if y_sent[i] == 'O':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                    if current_entity in entities_doc:
                        for j in current_indexes:
                            x_sent[j]['corp_entity_maj'] = entities_doc[current_entity].most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []

    for key in entities_corp.keys():
        entities_corp[key] = Counter(entities_corp[key])

    for x_doc, y_doc in zip(x_train_docs, y_train_docs_predicted):
        for x_sent, y_sent in zip(x_doc, y_doc):
            current_indexes = []
            for i in range(len(x_sent)):
                if y_sent[i][0] == 'S':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                    for j in current_indexes:
                        x_sent[j]['corp_entity_maj'] = entities_corp[current_entity].most_common(1)[0][0]
                    current_indexes = []
                    current_entity = ''
                if y_sent[i][0] == 'B':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                if y_sent[i][0] == 'I':
                    current_entity += ' ' + x_sent[i]['word.lower()']
                    current_indexes.append(i)
                if y_sent[i][0] == 'E':
                    current_entity += ' ' + x_sent[i]['word.lower()']
                    current_indexes.append(i)
                    if current_entity in entities_corp:
                        for j in current_indexes:
                            x_sent[j]['corp_entity_maj'] = entities_corp[current_entity].most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []
                if y_sent[i] == 'O':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                    if current_entity in entities_corp:
                        for j in current_indexes:
                            x_sent[j]['corp_entity_maj'] = entities_corp[current_entity].most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []


def add_super_features(x_train_docs, y_train_docs_predicted):
    super_corp = dict()
    current_entity = ''

    for x_doc, y_doc in zip(x_train_docs, y_train_docs_predicted):
        super_doc = dict()
        for x_sent, y_sent in zip(x_doc, y_doc):
            for x_token, y_token in zip(x_sent, y_sent):
                token = x_token['word.lower()']
                if y_token[0] == 'S':
                    current_entity = token
                    if current_entity in super_doc:
                        super_doc[current_entity].append(y_token[2:])
                    else:
                        super_doc[current_entity] = [y_token[2:]]
                    if current_entity in super_corp:
                        super_corp[current_entity].append(y_token[2:])
                    else:
                        super_corp[current_entity] = [y_token[2:]]
                if y_token[0] == 'B':
                    current_entity = token
                if y_token[0] == 'I':
                    current_entity += ' ' + token
                if y_token[0] == 'E':
                    current_entity += ' ' + token
                    if current_entity in super_doc:
                        super_doc[current_entity].append(y_token[2:])
                    else:
                        super_doc[current_entity] = [y_token[2:]]
                    if current_entity in super_corp:
                        super_corp[current_entity].append(y_token[2:])
                    else:
                        super_corp[current_entity] = [y_token[2:]]
                    current_entity = ''
                if y_token == 'O':
                    current_entity = ''
        for key in super_doc.keys():
            super_doc[key] = Counter(super_doc[key])
        for x_sent, y_sent in zip(x_doc, y_doc):
            current_indexes = []
            for i in range(len(x_sent)):
                if y_sent[i][0] == 'S':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                    current_counter = Counter()
                    for key in super_doc.keys():
                        if current_entity in key and len(current_entity) != len(key):
                            current_counter += super_doc[key]
                    if len(current_counter.keys()) != 0:
                        for j in current_indexes:
                            x_sent[j]['doc_super_maj'] = current_counter.most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []
                if y_sent[i][0] == 'B':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                if y_sent[i][0] == 'I':
                    current_entity += ' ' + x_sent[i]['word.lower()']
                    current_indexes.append(i)
                if y_sent[i][0] == 'E':
                    current_entity += ' ' + x_sent[i]['word.lower()']
                    current_indexes.append(i)
                    current_counter = Counter()
                    for key in super_doc.keys():
                        if current_entity in key and len(current_entity) != len(key):
                            current_counter += super_doc[key]
                    if len(current_counter.keys()) != 0:
                        for j in current_indexes:
                            x_sent[j]['doc_super_maj'] = current_counter.most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []
                if y_sent[i] == 'O':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                    current_counter = Counter()
                    for key in super_doc.keys():
                        if current_entity in key and len(current_entity) != len(key):
                            current_counter += super_doc[key]
                    if len(current_counter.keys()) != 0:
                        for j in current_indexes:
                            x_sent[j]['doc_super_maj'] = current_counter.most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []

    for key in super_corp.keys():
        super_corp[key] = Counter(super_corp[key])
    for x_doc, y_doc in zip(x_train_docs, y_train_docs_predicted):
        for x_sent, y_sent in zip(x_doc, y_doc):
            current_indexes = []
            for i in range(len(x_sent)):
                if y_sent[i][0] == 'S':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                    current_counter = Counter()
                    for key in super_corp.keys():
                        if current_entity in key and len(current_entity) != len(key):
                            current_counter += super_corp[key]
                    if len(current_counter.keys()) != 0:
                        for j in current_indexes:
                            x_sent[j]['corp_super_maj'] = current_counter.most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []
                if y_sent[i][0] == 'B':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                if y_sent[i][0] == 'I':
                    current_entity += ' ' + x_sent[i]['word.lower()']
                    current_indexes.append(i)
                if y_sent[i][0] == 'E':
                    current_entity += ' ' + x_sent[i]['word.lower()']
                    current_indexes.append(i)
                    current_counter = Counter()
                    for key in super_corp.keys():
                        if current_entity in key and len(current_entity) != len(key):
                            current_counter += super_corp[key]
                    if len(current_counter.keys()) != 0:
                        for j in current_indexes:
                            x_sent[j]['corp_super_maj'] = current_counter.most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []
                if y_sent[i] == 'O':
                    current_entity = x_sent[i]['word.lower()']
                    current_indexes = [i]
                    current_counter = Counter()
                    for key in super_corp.keys():
                        if current_entity in key and len(current_entity) != len(key):
                            current_counter += super_corp[key]
                    if len(current_counter.keys()) != 0:
                        for j in current_indexes:
                            x_sent[j]['corp_super_maj'] = current_counter.most_common(1)[0][0]
                    current_entity = ''
                    current_indexes = []
                    
def word2features(sent, i):
    word = sent[i][0]
    postag = sent[i][1]
    chunktag = sent[i][2]

    features = {
        'bias': 1.0,
        'word': word,
        'word.lower()': word.lower(),
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.isalpha()': word.isalpha(),
        'postag': postag,
        'postag[:2]': postag[:2],
        'chunktag': chunktag,
        'chunktag[:2]': chunktag[:2],
    }
    if i > 0:
        word1 = sent[i - 1][0]
        postag1 = sent[i - 1][1]
        chunktag1 = sent[i - 1][2]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
            '-1:word.isalpha()': word1.isalpha(),
            '-1:postag': postag1,
            '-1:postag[:2]': postag1[:2],
            '-1:chunktag': chunktag1,
            '-1:chunktag[:2]': chunktag1[:2],
        })
    else:
        features['BOS'] = True
    
    if i > 1:
        word2 = sent[i - 2][0]
        postag2 = sent[i - 2][1]
        chunktag2 = sent[i - 2][2]
        features.update({
            '-2:word.lower()': word2.lower(),
            '-2:word.istitle()': word2.istitle(),
            '-2:word.isupper()': word2.isupper(),
            '-2:word.isalpha()': word2.isalpha(),
            '-2:postag': postag2,
            '-2:postag[:2]': postag2[:2],
            '-2:chunktag': chunktag2,
            '-2:chunktag[:2]': chunktag2[:2],
        })

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]
        postag1 = sent[i + 1][1]
        chunktag1 = sent[i + 1][2]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
            '+1:word.isalpha()': word1.isalpha(),
            '+1:postag': postag1,
            '+1:postag[:2]': postag1[:2],
            '+1:chunktag': chunktag1,
            '+1:chunktag[:2]': chunktag1[:2],
        })
    else:
        features['EOS'] = True
        
    if i < len(sent) - 2:
        word2 = sent[i + 2][0]
        postag2 = sent[i + 2][1]
        chunktag2 = sent[i + 2][2]
        features.update({
            '+2:word.lower()': word2.lower(),
            '+2:word.istitle()': word2.istitle(),
            '+2:word.isupper()': word2.isupper(),
            '+2:word.isalpha()': word2.isalpha(),
            '+2:postag': postag2,
            '+2:postag[:2]': postag2[:2],
            '+2:chunktag': chunktag2,
            '+2:chunktag[:2]': chunktag2[:2],
        })

    return features


def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def sent2labels(sent):
    return [label for token, postag, label in sent]

def sent2tokens(sent):
    return [token for token, postag, label in sent]