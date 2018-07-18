import math
import os
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
#import ipdb
from tensorflow.models.rnn import rnn_cell
import tensorflow.python.platform
import sequence

### Paramameters
n_epochs=1000
batch_size=100
dim_embed=256
dim_ctx=512
dim_hidden=256
ctx_shape=[490,512]
train_ix=25078
valid_ix=26732

#pretrained_model_path='./model/model-8'
annotation_path='/mnt/ssd/jykim/annotation.pkl'
feat_path='/mnt/ssd/jykim/feat_image.npy'
model_path='/mnt/hdd/users/jykim/video_description/model/'
###


def preProBuildWordVocab(sentence_iterator, word_count_threshold=3):

    print 'preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold,)
    word_counts = {}
    nsents = 0

    for sent in sentence_iterator:
        nsents += 1

        for w in sent.lower().split(' '):
            word_counts[w] = word_counts.get(w, 0) + 1

    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print 'filtered words from %d to %d' % (len(word_counts), len(vocab))

    ixtoword = {}
    ixtoword[0] = '.'
    wordtoix = {}
    wordtoix['#START#'] = 0

    ix = 1
    for w in vocab:
        wordtoix[w] = ix
        ixtoword[ix] = w
        ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector)
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector)

    return wordtoix, ixtoword, bias_init_vector


def initialize_data():
    annotation_data = pd.read_pickle(annotation_path)
    captions = annotation_data['caption'].values

    n_captions = len(captions)

    for i in range(n_captions):
        captions[i] = captions[i].replace('.', ' .')
        captions[i] = captions[i].replace(',', ' ,')
        captions[i] = captions[i].replace('!', ' .')
        captions[i] = captions[i].replace('?', ' .')
        captions[i] = captions[i].replace('"', '')
        captions[i] = captions[i].replace("'", '')

        if captions[i][-1] != '.':
            captions[i] += ' .'

    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

    feats = np.load(feat_path)
    train_feats = feats[0:1501]
    valid_feats = feats[1501:1601]
    test_feats = feats[1601:]

    return wordtoix, ixtoword, bias_init_vector, train_feats, valid_feats, test_feats

wordtoix,ixtoword,bias_init_vector,train_feats, valid_feats, test_feats = initialize_data()

class Caption_Generator():

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, n_words, dim_embed, dim_ctx, dim_hidden, n_lstm_steps, batch_size=200, ctx_shape=[490,512], bias_init_vector=None):
        #initialize params of model
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_ctx = dim_ctx
        self.dim_hidden = dim_hidden
        self.ctx_shape = ctx_shape
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -1.0, 1.0), name='Wemb')

        #for initializing LSTM
        self.init_hidden_W = self.init_weight(dim_ctx, dim_hidden, name='init_hidden_W')
        self.init_hidden_b = self.init_bias(dim_hidden, name='init_hidden_b')
        self.init_memory_W = self.init_weight(dim_ctx, dim_hidden, name='init_memory_W')
        self.init_memory_b = self.init_bias(dim_hidden, name='init_memory_b')

        #for main LSTM
        self.lstm_W = self.init_weight(dim_embed, dim_hidden*4, name='lstm_W') #LSTM input to hidden
        self.lstm_U = self.init_weight(dim_hidden, dim_hidden*4, name='lstm_U') #LSTM hidden to hidden
        self.lstm_b = self.init_bias(dim_hidden*4, name='lstm_b') #LSTM bias

        self.image_encode_W = self.init_weight(dim_ctx, dim_hidden*4, name='image_encode_W') #context to LSTM

        #for attention model
        self.image_att_W = self.init_weight(dim_ctx, dim_ctx, name='image_att_W')
        self.hidden_att_W = self.init_weight(dim_hidden, dim_ctx, name='hidden_att_W')
        self.pre_att_b = self.init_bias(dim_ctx, name='pre_att_b')

        self.att_W = self.init_weight(dim_ctx, 1, name='att_W')
        self.att_b = self.init_bias(1, name='att_b')

        #for decoding NN
        self.decode_lstm_W = self.init_weight(dim_hidden, dim_embed, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(dim_embed, name='decode_lstm_b')
        self.decode_word_W = self.init_weight(dim_embed, n_words, name='decode_word_W')

        if bias_init_vector is not None:
            self.decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_word_b')
        else:
            self.decode_word_b = self.init_bias(n_words, name='decode_word_b')

    def get_initial_lstm(self, mean_context):
        #initialize memory and hidden unit of lstm
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)

        return initial_hidden, initial_memory

    def build_model(self):
        #build overall architecture
        context = tf.placeholder("float32", [self.batch_size, self.ctx_shape[0], self.ctx_shape[1]])
        sentence = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps])

        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))

        context_flat = tf.reshape(context, [-1, self.dim_ctx])
        context_encode = tf.matmul(context_flat, self.image_att_W)
        context_encode = tf.reshape(context_encode, [-1, ctx_shape[0], ctx_shape[1]]) #(batch_size, 490, 512)

        loss = 0.0

        for ind in range(self.n_lstm_steps):
            if ind == 0:
                word_emb = tf.zeros([self.batch_size, self.dim_embed])
            else:
                tf.get_variable_scope().reuse_variables()
                with tf.device("/cpu:0"):
                    word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:, ind-1])
            x_t = tf.matmul(word_emb, self.lstm_W) +self.lstm_b

            labels = tf.expand_dims(sentence[:, ind], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense( concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

            context_encode = context_encode + \
                tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + \
                self.pre_att_b

            context_encode = tf.nn.tanh(context_encode)

            context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx])
            alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b
            alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]]) #(batch_size, 490)
            alpha = tf.nn.softmax(alpha)

            weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)
            i, f, o, new_c = tf.split(1, 4, lstm_preactive)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f * c + i * new_c
            h = o * tf.nn.tanh(c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)
            logits = tf.nn.dropout(logits, 0.5)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            cross_entropy = cross_entropy * mask[:, ind]

            current_loss = tf.reduce_sum(cross_entropy)
            loss = loss + current_loss

        loss = loss / tf.reduce_sum(mask)
        return loss, context, sentence, mask

    def build_generator(self, maxlen):
        context = tf.placeholder("float32", [1, self.ctx_shape[0], self.ctx_shape[1]])
        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))

        context_encode = tf.matmul(tf.squeeze(context), self.image_att_W)
        generated_words = []
        logit_list = []
        alpha_list = []
        word_emb = tf.zeros([1, self.dim_embed])
        for ind in range(maxlen):
            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b
            context_encode = context_encode + tf.matmul(h, self.hidden_att_W) + self.pre_att_b
            context_encode = tf.nn.tanh(context_encode)

            alpha = tf.matmul(context_encode, self.att_W) + self.att_b
            alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]])
            alpha = tf.nn.softmax(alpha)

            alpha = tf.reshape(alpha, (ctx_shape[0], -1))
            alpha_list.append(alpha)

            weighted_context = tf.reduce_sum(tf.squeeze(context) * alpha, 0)
            weighted_context = tf.expand_dims(weighted_context, 0)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)

            i, f, o, new_c = tf.split(1, 4, lstm_preactive)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f * c + i * new_c
            h = o * tf.nn.tanh(c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b

            max_prob_word = tf.argmax(logit_words, 1)

            with tf.device("/cpu:0"):
                word_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)

            generated_words.append(max_prob_word)
            logit_list.append(logit_words)

        return context, generated_words, logit_list, alpha_list


def train(pretrained_model_path, feats=train_feats,wordtoix=wordtoix,ixtoword=ixtoword,bias_init_vector=bias_init_vector): #load pretrained params

    annotation_data = pd.read_pickle(annotation_path)
    annotation_data = annotation_data.ix[0:train_ix]
    captions = annotation_data['caption'].values

    n_captions = len(captions)

    for i in range(n_captions):
        captions[i] = captions[i].replace('.', ' .')
        captions[i] = captions[i].replace(',', ' ,')
        captions[i] = captions[i].replace('!', ' .')
        captions[i] = captions[i].replace('?', ' .')
        captions[i] = captions[i].replace("'", '')
        captions[i] = captions[i].replace('"', '')

        if captions[i][-1] != '.':
            captions[i] += ' .'

    learning_rate=0.001
    n_words = len(wordtoix)

    maxlen = np.max( map(lambda x: len(x.split(' ')), captions) )

    sess = tf.InteractiveSession()

    caption_generator = Caption_Generator(n_words=n_words,dim_embed=dim_embed,dim_ctx=dim_ctx,dim_hidden=dim_hidden,n_lstm_steps=maxlen+1,batch_size=batch_size,ctx_shape=ctx_shape,bias_init_vector=bias_init_vector)

    loss, context, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()
    if pretrained_model_path is not None:
        print "Starting with pretrained model"
        saver.restore(sess, pretrained_model_path)

    index = list(annotation_data.index)
    np.random.shuffle(index)
    annotation_data = annotation_data.ix[index]

    captions = annotation_data['caption'].values
    video_id = annotation_data['video_id'].values

    for epoch in range(n_epochs):
        batch_num=0
        for start, end in zip(range(0, len(captions), batch_size), range(batch_size, len(captions), batch_size)):

            batch_num += 1

            current_feats = feats[video_id[start:end]]

            current_captions = captions[start:end]
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions)

            current_captions_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)

            current_mask_matrix = np.zeros((current_captions_matrix.shape[0], current_captions_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_captions_matrix ))

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={context:current_feats, sentence:current_captions_matrix, mask:current_mask_matrix})

            print "Epoch %d, minibatch %d, Current Cost: %f" % (epoch+1, batch_num, loss_value)
        saver.save(sess, os.path.join(model_path, 'train_2'), global_step=epoch+1)

def test(test_feat, model_path, wordtoix=wordtoix, ixtoword=ixtoword, bias_init_vector=bias_init_vector, maxlen=20):

    n_words = len(wordtoix)

    sess = tf.InteractiveSession()

    caption_generator = Caption_Generator(n_words=n_words,dim_embed=dim_embed,dim_ctx=dim_ctx,dim_hidden=dim_hidden,n_lstm_steps=maxlen,batch_size=batch_size,ctx_shape=ctx_shape)

    context, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=maxlen)
    saver=tf.train.Saver()
    saver.restore(sess, model_path)

    generated_word_index = sess.run(generated_words, feed_dict={context:test_feat})
    alpha_list_val = sess.run(alpha_list, feed_dict={context:test_feat})
    generated_words = [ixtoword[x[0]] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.')+1

    generated_words = generated_words[:punctuation]
    alpha_list_val = alpha_list_val[:punctuation]
    return generated_words, alpha_list_val
