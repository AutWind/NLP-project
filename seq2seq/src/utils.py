import numpy as np


# 定义其他的函数
def pad_batch(batch, char_to_int):
    sequence_length = [len(sequence) for sequence in batch]
    max_length = max(sequence_length)

    new_batch = [sequence + [char_to_int["<PAD>"]] * (max_length - len(sequence)) for sequence in batch]

    return sequence_length, max_length, new_batch


def next_batch(source, target, batch_size, source_char_to_int, target_char_to_int):
    num_batches = len(source) // batch_size
    for i in range(num_batches):
        source_batch = source[i * batch_size: (i + 1) * batch_size]
        target_batch = target[i * batch_size: (i + 1) * batch_size]

        source_sequence_length, source_max_length, new_source_batch = pad_batch(source_batch, source_char_to_int)
        target_sequence_length, target_max_length, new_target_batch = pad_batch(target_batch, target_char_to_int)

        yield dict(source_batch=np.array(new_source_batch), target_batch=np.array(new_target_batch),
                   source_sequence_length=np.array(source_sequence_length),
                   target_sequence_length=np.array(target_sequence_length),
                   target_max_length=target_max_length)