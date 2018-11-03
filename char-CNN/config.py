"""
配置模型和训练时的所需参数
"""


class TrainingConfig(object):
    epoches = 10
    evaluate_every = 100
    checkpoint_every = 100


class ModelConfig(object):
    conv_layers = [[256, 7, 3],
                   [256, 7, 3],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, None],
                   [256, 3, 3]]
    #     conv_layers = [[1024, 7, 3],
    #                   [1024, 7, 3],
    #                   [1024, 3, None],
    #                   [1024, 3, None],
    #                   [1024, 3, None],
    #                   [1024, 3, 3]]

    #     fc_layers = [2048, 2048]
    fc_layers = [1024, 1024]
    dropout_keep_prob = 0.5
    learning_rate = 0.0005


class Config(object):
    sequenceLength = 1014
    batch_size = 128

    data_source = "../data/data_v1/500_middle_math_skills.csv"

    num_classes = 55

    training = TrainingConfig()

    model = ModelConfig()


config = Config()