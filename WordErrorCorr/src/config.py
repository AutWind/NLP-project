# 配置参数
class ModelConfig(object):
    encoder_hidden_layers = [50, 50]
    decoder_hidden_layers = [50, 50]
    dropout_prob = 0.5
    encoder_embedding_size = 15
    decoder_embedding_size = 15


class TrainConfig(object):
    epochs = 10
    every_checkpoint = 100
    learning_rate = 0.01
    max_grad_norm = 3


class Config(object):
    batch_size = 128
    infer_prob = 0.2

    source_path = "data/letters_source.txt"
    target_path = "data/letters_target.txt"

    train = TrainConfig()
    model = ModelConfig()


