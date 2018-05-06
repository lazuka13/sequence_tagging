from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.config import Config
from format_predictions import format_predictions

import numpy as np


def main(pretrained):
    config = Config(dir_output='./results/train_first/')
    dev_predictions_file = './data/predictions/formatted_dev_predictions.npy'

    model = NERModel(config)
    model.build()

    dev = CoNLLDataset(config.filename_dev, config.processing_word,
                       config.processing_tag, config.max_iter)
    train = CoNLLDataset(config.filename_train, config.processing_word,
                         config.processing_tag, config.max_iter)

    if pretrained:
        model.restore_session(config.dir_model)
    else:
        model.train(train, dev)

    print('Prediction stage!')
    dev_predictions = model.predict_test(dev)
    print('Formatting stage!')
    formatted_predictions = format_predictions(dev_predictions, 'dev', config)
    print('Saving stage!')
    np.save(dev_predictions_file, formatted_predictions)

    test = CoNLLDataset(config.filename_test, config.processing_word,
                         config.processing_tag, config.max_iter)
    model.evaluate(test)


if __name__ == "__main__":
    main(False)
