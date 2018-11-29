import tensorflow as tf
import string

BATCH_SIZE = 128
MAX_WORDS_IN_REVIEW = 100  # Maximum length of a review to consider
EMBEDDING_SIZE = 50  # Dimensions for each word vector
NUM_HIDDEN = 128
num_LSTM_layers = 3

stop_words = set({'ourselves', 'hers', 'between', 'yourself', 'again',
                  'there', 'about', 'once', 'during', 'out', 'very', 'having',
                  'with', 'they', 'own', 'an', 'be', 'some', 'for', 'do', 'its',
                  'yours', 'such', 'into', 'of', 'most', 'itself', 'other',
                  'off', 'is', 's', 'am', 'or', 'who', 'as', 'from', 'him',
                  'each', 'the', 'themselves', 'below', 'are', 'we',
                  'these', 'your', 'his', 'through', 'don', 'me', 'were',
                  'her', 'more', 'himself', 'this', 'down', 'should', 'our',
                  'their', 'while', 'above', 'both', 'up', 'to', 'ours', 'had',
                  'she', 'all', 'no', 'when', 'at', 'any', 'before', 'them',
                  'same', 'and', 'been', 'have', 'in', 'will', 'on', 'does',
                  'yourselves', 'then', 'that', 'because', 'what', 'over',
                  'why', 'so', 'can', 'did', 'not', 'now', 'under', 'he', 'you',
                  'herself', 'has', 'just', 'where', 'too', 'only', 'myself',
                  'which', 'those', 'i', 'after', 'few', 'whom', 't', 'being',
                  'if', 'theirs', 'my', 'against', 'a', 'by', 'doing', 'it',
                  'how', 'further', 'was', 'here', 'than'})

def create_lstm_cell():
    lstm = tf.contrib.rnn.BasicLSTMCell(NUM_HIDDEN, forget_bias=0.5)
    # lstm = tf.contrib.rnn.GRUCell(NUM_HIDDEN)
    # lstm = tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=0.6)
    return lstm

def create_gru_cell():
    gru = tf.contrib.rnn.GRUCell(NUM_HIDDEN)
    gru = tf.contrib.rnn.DropoutWrapper(gru, output_keep_prob=0.6)
    return gru

def preprocess(review):
    """
    Apply preprocessing to a single review. You can do anything here that is manipulation
    at a string level, e.g.
        - removing stop words
        - stripping/adding punctuation
        - changing case
        - word find/replace
    RETURN: the preprocessed review in string form.
    """
    # https://machinelearningmastery.com/clean-text-machine-learning-python/
    # split into tokens by white space
    tokens = review.split()
    
    # convert to lower case
    lower_tokens = []
    for t in tokens:
        lower_tokens.append(t.lower())
    del tokens

    tokens = lower_tokens

    # remove punctuation from each token
    table = str.maketrans('', '', string.punctuation)
    tokens = [w.translate(table) for w in tokens]

    # remove remaining tokens that are not completely alphabetic
    tokens = [word for word in tokens if word.isalpha()]

    # filter by stop words
    tokens = [w for w in tokens if not w in stop_words]

    return tokens

def define_graph():
    """
    Implement your model here. You will need to define placeholders, for the input and labels,
    Note that the input is not strings of words, but the strings after the embedding lookup
    has been applied (i.e. arrays of floats).

    In all cases this code will be called by an unaltered runner.py. You should read this
    file and ensure your code here is compatible.

    Consult the assignment specification for details of which parts of the TF API are
    permitted for use in this function.

    You must return, in the following order, the placeholders/tensors for;
    RETURNS: input, labels, optimizer, accuracy and loss
    """

    # Define input data
    input_data = tf.placeholder(tf.float32, [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE], name='input_data')
    # [BATCH_SIZE, MAX_WORDS_IN_REVIEW, EMBEDDING_SIZE]
    labels = tf.placeholder(tf.float32, [BATCH_SIZE, 2], name='labels')
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.85)
    dropout_keep_prob = tf.placeholder_with_default(0.5, shape=())

    # RNN layers
    cell = tf.contrib.rnn.MultiRNNCell([create_lstm_cell() for cell_num in range(num_LSTM_layers)])
    outputs, state = tf.nn.dynamic_rnn(cell, input_data, dtype=tf.float32)
    print(outputs.get_shape())
    outputs = tf.reduce_mean(outputs, axis=1)

    # # Fully connected layer
    # weight = tf.Variable(tf.truncated_normal(shape=[NUM_HIDDEN, 2], stddev=0.001))
    # bias = tf.Variable(tf.constant(0.01, shape=[2]))
    # logits = tf.matmul(outputs, weight) + bias
    # prediction = tf.nn.softmax(logits)

    weight = tf.Variable(tf.random_uniform(shape=[NUM_HIDDEN, 2],maxval = 0.001))
    bias = tf.Variable(tf.random_uniform([2],maxval=0.001))
    logits = tf.matmul(outputs, weight) + bias
    prediction = tf.nn.softmax(logits)

    # accuracy
    predict_labels = tf.argmax(logits, 1) # Apply argmax on the rows, axis = 0 will be columns
    real_labels = tf.argmax(labels, 1)
    correct_prediction = tf.equal(predict_labels, real_labels) # Compares to see if prediction is same as real_labels
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, dtype=tf.float32), name="accuracy")
    
    # loss
    xentropy = tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels)
    loss = tf.reduce_mean(xentropy, name="loss")
    
    # optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate= 0.001).minimize(loss)
    return input_data, labels, dropout_keep_prob, optimizer, accuracy, loss

# Testing pre processing to see it works
# def test():
    # texts = "Once again Mr. Costner has dragged out a movie for far longer than necessary. Aside from the terrific sea rescue sequences, of which there are very few I just did not care about any of the characters. Most of us have ghosts in the closet, and Costner's character are realized early on, and then forgotten until much later, by which time I did not care. The character we should really care about is a very cocky, overconfident Ashton Kutcher. The problem is he comes off as kid who thinks he's better than anyone else around him and shows no signs of a cluttered closet. His only obstacle appears to be winning over Costner. Finally when we are well past the half way point of this stinker, Costner tells us all about Kutcher's ghosts. We are told why Kutcher is driven to be the best with no prior inkling or foreshadowing. No magic here, it was all I could do to keep from turning it off an hour in."
    # print('before processing')
    # print(texts)

    # print('after processing')
    # print(preprocess(texts))

    
# if __name__ == "__main__":
#     test()