from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config

from sklearn.model_selection import KFold
import numpy as np
from model.data_utils import minibatches
import tensorflow as tf


def main():
    config = Config()
    predictions_file = './data/predictions'

    kf = KFold(n_splits=5)

    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    train = np.array([el for el in train])
    predictions = [0 for _ in train]

    for train_ids, evaluate_ids in kf.split(train):
        train_dataset = train[train_ids]
        evaluate_dataset = train[evaluate_ids]
        tf.reset_default_graph()
        model = NERModel(config)
        model.build()
        model.train(train_dataset, evaluate_dataset)
        for id, tags in zip(evaluate_ids, model.predict_test(evaluate_dataset)):
            predictions[id] = tags
        model.close_session()

    predictions = np.array(predictions)
    np.save(predictions_file, predictions)


if __name__ == "__main__":
    main()
