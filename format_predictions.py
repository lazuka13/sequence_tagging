from utils import *


def format_predictions(data, dataset_type, config):
    file_name = None
    if dataset_type == 'dev':
        file_name = 'eng.testa.dev.txt'
    elif dataset_type == 'train':
        file_name = 'eng.train.txt'
    elif dataset_type == 'test':
        file_name = 'eng.testb.test.txt'

    x_train_docs, y_train_docs = docs_from_dataset('./data/conll', file_name,
                                                   ('words', 'pos', 'chunk', 'ne'),
                                                   ['words', 'pos', 'chunk'], sent2features)

    index = 0
    predictions_as_docs = []
    for doc in y_train_docs:
        new_pred_doc = []
        for sent in doc:
            new_pred_doc.append(data[index])
            index += 1
        predictions_as_docs.append(new_pred_doc)

    add_token_features(x_train_docs, predictions_as_docs)
    add_entity_features(x_train_docs, predictions_as_docs)
    add_super_features(x_train_docs, predictions_as_docs)

    from sklearn.preprocessing import OneHotEncoder

    encoder = LabelEncoder()

    encoder.get('O')

    def get_features(x_data):
        return [
            x_data.get('doc_token_maj', 'O'),
            x_data.get('corp_token_maj', 'O'),
            x_data.get('doc_entity_maj', 'O'),
            x_data.get('doc_entity_maj', 'O'),
            x_data.get('corp_super_maj', 'O'),
            x_data.get('corp_super_maj', 'O')
        ]

    dataset_formatted = []

    for x_doc, y_doc in zip(x_train_docs, y_train_docs):
        for x_sent, y_sent in zip(x_doc, y_doc):
            words = []
            tags = []
            features = []
            for x_data, y_token in zip(x_sent, y_sent):
                additional_features = get_features(x_data)
                words.append(config.processing_word(x_data['word']))
                tags.append(config.processing_tag(y_token))
                features.append([encoder.get(tag) for tag in additional_features])
            dataset_formatted.append([words, features, tags])

    only_features = [sent[1] for sent in dataset_formatted]
    only_features_flat = []

    for sent_features in only_features:
        only_features_flat += sent_features

    enc = OneHotEncoder()
    enc.fit(only_features_flat)

    for sent in dataset_formatted:
        sent[1] = [enc.transform([word_features]).toarray()[0].tolist() for word_features in sent[1]]

    return dataset_formatted
