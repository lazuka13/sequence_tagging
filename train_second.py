import numpy as np

from model.ner_model_second import NERModel as NERModel2
from model.config import Config


def main():
    config = Config('./results/train_second/')
    train_data_file = './data/predictions/formatted_train_predictions.npy'
    dev_data_file = './data/predictions/formatted_dev_predictions.npy'
    train_data = np.load(train_data_file)
    dev_data = np.load(dev_data_file)

    model = NERModel2(config)
    model.build()
    model.train(train_data, dev_data)


if __name__ == "__main__":
    main()
