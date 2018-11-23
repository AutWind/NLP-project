# 配置参数


class TrainingConfig(object):
    epoches = 100
    evaluate_every = 100
    checkpoint_every = 100
    learning_rate = 1e-3


class ModelConfig(object):
    embedding_size = 128
    num_filters = 128

    filter_sizes = [3, 4, 5]
    dropout_keep_prob = 0.5
    l2_reg_lambda = 0.0


class Config(object):
    sequence_length = 128
    batch_size = 64

    test_data_source = "../data/data_v2/token_seven_test.csv"
    train_data_source = "../data/data_v2/token_seven_train.csv"
    eval_data_source = "../data/data_v2/token_seven_eval.csv"
    stop_word_source = "../data/data_v2/stop_words.txt"

    num_classes = 60
    l2_reg_lambda = 0.0
    training = TrainingConfig()

    model = ModelConfig()


config = Config()