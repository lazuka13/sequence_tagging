from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from format_predictions import format_predictions

from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np


def main():
    config = Config('./results/train_folds/')
    train_predictions_file = './data/predictions/formatted_train_predictions.npy'

    kf = KFold(n_splits=5)

    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    train = np.array([el for el in train])
    predictions = [0 for _ in train]

    for train_ids, evaluate_ids in kf.split(train):
        train_dataset = train[train_ids]
        evaluate_dataset = train[evaluate_ids]
        tf.reset_default_graph()
        config = Config('./results/train_folds/')
        model = NERModel(config)
        model.build()
        model.train(train_dataset, evaluate_dataset)
        for id, tags in zip(evaluate_ids, model.predict_test(evaluate_dataset)):
            predictions[id] = tags
        model.close_session()

    predictions = np.array(predictions)
    formatted_predictions = format_predictions(predictions, 'train', config)
    np.save(train_predictions_file, formatted_predictions)


if __name__ == "__main__":
    main()
