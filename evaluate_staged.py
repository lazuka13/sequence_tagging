from model.data_utils import CoNLLDataset
from model.ner_model import NERModel
from model.ner_model_second import NERModel as NERModel2
from model.config import Config
from format_predictions import format_predictions
import tensorflow as tf


def main():
    # Предсказания моделью первого уровня #
    config_first = Config(dir_output='./results/train_first/')
    model = NERModel(config_first)
    model.build()
    model.restore_session(config_first.dir_model)
    test = CoNLLDataset(config_first.filename_test, config_first.processing_word,
                        config_first.processing_tag, config_first.max_iter)

    print()
    print('Predicting first stage!')
    model.evaluate(test)
    print()

    test_predictions = model.predict_test(test)
    formatted_predictions = format_predictions(test_predictions, 'test', config_first)

    # Предсказания моделью второго уровня #
    tf.reset_default_graph()
    config_second = Config(dir_output='./results/train_second/')
    model = NERModel2(config_second)
    model.build()
    model.restore_session(config_second.dir_model)

    print()
    print('Predicting second stage!')
    model.evaluate(formatted_predictions)
    print()


if __name__ == "__main__":
    main()
