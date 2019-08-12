#encoding=utf8
import zipfile
import tensorflow as tf
import collections
import numpy as np
import random



def readData(filename):
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

def buildDictionary(words, vocabulary_size):
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(vocabulary_size - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)

    unk_count = 0
    data = [dictionary[word] if word in dictionary else 0 for word in words]

    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reverse_dictionary


def generate_batch(data, batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1)
    for i in range(batch_size // num_skips):  # i取值0,1,2,3
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]

        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels

def train_w2v(dataFile):
    words = readData(dataFile)
    vocabulary_size = len(set(words))
    print('Data size', vocabulary_size)

    data, count, dictionary, reverse_dictionary = buildDictionary(words)

    del words  # Hint to reduce memory.
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

    data_index = 0
    batch, labels = generate_batch(batch_size=8, num_skips=4, skip_window=2)
    for i in range(8):
        print(batch[i], reverse_dictionary[batch[i]],
              '->', labels[i, 0], reverse_dictionary[labels[i, 0]])

    batch_size = 128
    embedding_size = 300  # dimension of the embedding vector
    skip_window = 2  # how many words to consider to left and right
    num_skips = 4  # how many times to reuse an input to generate a label

    # we choose random validation dataset to sample nearest neighbors
    # here, we limit the validation samples to the words that have a low
    # numeric ID, which are also the most frequently occurring words
    valid_size = 16  # size of random set of words to evaluate similarity on
    valid_window = 100  # only pick development samples from the first 'valid_window' words
    valid_examples = np.random.choice(valid_window, valid_size, replace=False)
    num_sampled = 64  # number of negative examples to sample

    # create computation graph
    graph = tf.Graph()

    with graph.as_default():
        # input data
        train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # operations and variables
        # look up embeddings for inputs
        embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # construct the variables for the NCE loss
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        # compute the average NCE loss for the batch.
        # tf.nce_loss automatically draws a new sample of the negative labels each time we evaluate the loss.
        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nce_weights, biases=nce_biases,
                                             labels=train_labels, inputs=embed, num_sampled=num_sampled,
                                             num_classes=vocabulary_size))

        # construct the SGD optimizer using a learning rate of 1.0
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

        # compute the cosine similarity between minibatch examples and all embeddings
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
        normalized_embeddings = embeddings / norm
        valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)


        # add variable initializer
        init = tf.initialize_all_variables()
    # 5
    num_steps = 1000

    with tf.Session(graph=graph) as session:
        # we must initialize all variables before using them
        init.run()
        print('initialized.')

        # loop through all training steps and keep track of loss
        average_loss = 0

        for step in range(num_steps):
            # generate a minibatch of training data
            batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
            feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

            # we perform a single update step by evaluating the optimizer operation (including it
            # in the list of returned values of session.run())
            _, loss_val, ncs_loss_ = session.run([optimizer, loss], feed_dict=feed_dict)
            average_loss += loss_val
            final_embeddings = normalized_embeddings.eval()
            print(final_embeddings)
            print("*" * 20)

    final_embeddings = normalized_embeddings.eval()
    print(final_embeddings)
    fp = open('vector.txt', 'w', encoding='utf8')
    for k, v in reverse_dictionary.items():
        t = tuple(final_embeddings[k])
        # t1=[str(i) for i in t]
        s = ''
        for i in t:
            i = str(i)
            s += i + " "

        fp.write(v + " " + s + "\n")

    fp.close()


def countSimilarity(word1, word2):
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)