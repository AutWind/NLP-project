
# 生成数据


class DataGen(object):

    def __init__(self, config):
        self.source_path = config.source_path
        self.target_path = config.target_path

        self.source_char_to_int = {}
        self.source_int_to_char = {}
        self.target_char_to_int = {}
        self.target_int_to_char = {}

        self.source_data = []
        self.target_data = []

    def read_data(self):
        with open(self.source_path, "r") as f:
            source_char_to_int, source_int_to_char, source_data = self.gen_vocab_dict(f.read())
        self.source_char_to_int = source_char_to_int
        self.source_int_to_char = source_int_to_char
        self.source_data = source_data

        with open(self.target_path, "r") as f:
            target_char_to_int, target_int_to_char, target_data = self.gen_vocab_dict(f.read(), True)
        self.target_char_to_int = target_char_to_int
        self.target_int_to_char = target_int_to_char
        self.target_data = target_data

    def gen_vocab_dict(self, string, is_target=False):
        special_words = ['<PAD>', '<UNK>', '<GO>', '<EOS>']
        vocab = list(set(string))
        vocab.remove("\n")
        vocab = special_words + vocab

        int_to_char = {index: char for index, char in enumerate(vocab)}
        char_to_int = {char: index for index, char in int_to_char.items()}

        word_list = string.strip().split("\n")
        if is_target:
            data = [[char_to_int.get(char, '<UNK>') for char in word] + [char_to_int['<EOS>']] for word in word_list]
        else:
            data = [[char_to_int.get(char, '<UNK>') for char in word] for word in word_list]
        return char_to_int, int_to_char, data


