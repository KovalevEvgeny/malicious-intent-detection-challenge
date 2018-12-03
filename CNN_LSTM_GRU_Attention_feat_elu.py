import time
start_time = time.time()
import warnings
warnings.filterwarnings('ignore')

import msgpack
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import re
np.random.seed(13)
import tensorflow as tf
tf.set_random_seed(seed=13)
from keras import initializers, regularizers, constraints, optimizers
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import MaxPooling1D, GlobalMaxPool1D, GlobalAvgPool1D, Dense, Input, Embedding, Dropout, CuDNNGRU, CuDNNLSTM, Conv1D, Bidirectional, concatenate
from keras.preprocessing import text as keras_text, sequence as keras_seq
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc
from keras import backend as K

import os
os.environ['OMP_NUM_THREADS'] = '8'


info = pd.read_csv('train_info.csv')
with open('train.msgpack', 'rb') as data_file:
    train = msgpack.unpack(data_file)
with open('test.msgpack', 'rb') as data_file:
    test = msgpack.unpack(data_file)
subm = pd.read_csv('sample_submission.csv')
train = pd.DataFrame(train)
test = pd.DataFrame(test)
train.columns = ['id', 'text']
test.columns = ['id', 'text']
y_train = np.array([1 if i == True else 0 for i in info.injection.values])

def add_features(data):
    eps = 1e-10
    data['text_decoded'] = data['text'].apply(lambda x: x.decode('ISO-8859-1'))
    data['length'] = data['text_decoded'].apply(lambda x: len(x))
    
    data['long'] = data['text_decoded'].apply(lambda x: 0 if len(x) <= 1000 else 1)
    
    data['digits'] = data['text_decoded'].apply(lambda x: sum(c.isdigit() for c in str(x)))
    data['digits2length'] = data['digits'] / (data['length'] + eps)
    
    data['isalpha'] = data['text_decoded'].apply(lambda x: sum(c.isalpha() for c in str(x)))
    data['isalpha2length'] = data['isalpha'] / (data['length'] + eps)
    
    data['islower'] = data['text_decoded'].apply(lambda x: sum(c.islower() for c in str(x)))
    data['islower2length'] = data['islower'] / (data['length'] + eps)
    
    data['isupper'] = data['text_decoded'].apply(lambda x: sum(c.isupper() for c in str(x)))
    data['isupper2length'] = data['isupper'] / (data['length'] + eps)
    
    data['isspace'] = data['text_decoded'].apply(lambda x: sum(c.isspace() for c in str(x)))
    data['isspace2length'] = data['isspace'] / (data['length'] + eps)
    
    data['istitle'] = data['text_decoded'].apply(lambda x: sum(c.istitle() for c in str(x)))
    data['istitle2length'] = data['istitle'] / (data['length'] + eps)
    
    data['some_bytes'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r"some bytes", x)) == 0 else 1)
    
    data['isascii'] = data['text_decoded'].apply(lambda x: 0 if max(ord(char) for char in x+' ') < 128 else 1)
    
    data['mail'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r"@\w+", x)) == 0 else 1)
    
    data['url'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r"http\w+", x)) == 0 else 1)
    
    data['tag'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r'<[^>]*>', x)) == 0 else 1)
    
    data['cookie'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r'cookie', x)) == 0 else 1)
    
    data['union'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r'union', x)) == 0 else 1)
    
    data['select'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r'select', x)) == 0 else 1)
    
    data['brackets'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r'\(\)\)', x)) == 0 else 1)
    
    data['fig_brackets'] = data['text_decoded'].apply(lambda x: 0 if len(re.findall(r"{", x)) + len(re.findall(r"}", x)) == 0 else 1)
    
    return data

train = add_features(train)
test = add_features(test)

features = [
        'long',
        'digits2length',
        'isalpha2length',
        'islower2length',
        'isupper2length',
        'isspace2length',
        'istitle2length',
        'some_bytes',
        'isascii',
        'mail',
        'url',
        'tag',
        'cookie',
        'union',
        'select',
        'brackets',
        'fig_brackets'
        ]

train_features = train[features].values
test_features = test[features].values

# define network parameters
maxlen = 512

list_sentences_train = train['text'].fillna("unknown").values
list_sentences_test = test['text'].fillna("unknown").values

tokenizer = keras_text.Tokenizer(char_level = True)
tokenizer.fit_on_texts(list(list_sentences_train))
# train data
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
X_train = keras_seq.pad_sequences(list_tokenized_train, maxlen=maxlen)
# test data
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
X_test = keras_seq.pad_sequences(list_tokenized_test, maxlen=maxlen)


weights_path = "weights_validation/cnn_lstm_gru_attention_feat_elu.h5"
best_weights_path = "weights_best/cnn_lstm_gru_attention_feat_elu/"

class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data
        self.max_score = 0
        self.not_better_count = 0

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=1)
            score = roc_auc_score(self.y_val, y_pred[:, 0])
            print("\n ROC-AUC - epoch: %d - score: %.6f \n" % (epoch+1, score))
            if (score > self.max_score):
                print("*** New High Score (previous: %.6f) ***\n" % self.max_score)
                model.save_weights(weights_path)
                self.max_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 3:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('he_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim


def build_model(features, embed_size=300, dropout=0.2, kernel_initializer='he_uniform', clipvalue=1.0):
    features_input = Input(shape=(features.shape[1],))
    inp = Input(shape=(maxlen, ))
    x = Embedding(len(tokenizer.word_counts) + 1, embed_size)(inp)
    x = Conv1D(filters=256, kernel_size=5, padding='same', kernel_initializer=kernel_initializer, activation='relu')(x)
    x = MaxPooling1D(pool_size=5)(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(CuDNNLSTM(300, kernel_initializer=kernel_initializer, return_sequences=True))(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(CuDNNGRU(300, kernel_initializer=kernel_initializer, return_sequences=True))(x)
    avg_pool = GlobalAvgPool1D()(x)
    max_pool = GlobalMaxPool1D()(x)
    att = Attention(102)(x)
    x = concatenate([avg_pool, max_pool, features_input, att])
    x = Dense(128, activation='elu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[inp, features_input], outputs=x)
    adam = optimizers.adam(clipvalue=clipvalue)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model

model = build_model(train_features)

batch_size = 256
epochs = 100
gc.collect()
K.clear_session()

num_folds = 10

predict = np.zeros((test.shape[0],1))

scores = []
oof_predict = np.zeros((train.shape[0],1))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=13)

cnt = 1
for train_index, val_index in kf.split(X_train):
    print('Starting fold #{}'.format(cnt))
    kfold_y_train, kfold_y_val = y_train[train_index], y_train[val_index]
    kfold_X_train = X_train[train_index]
    kfold_X_train_features = train_features[train_index]
    kfold_X_val = X_train[val_index]
    kfold_X_val_features = train_features[val_index]
    
    gc.collect()
    K.clear_session()
    
    model = build_model(train_features)
    
    roc_auc_val = RocAucEvaluation(validation_data=([kfold_X_val, kfold_X_val_features], kfold_y_val))
    
    model.fit([kfold_X_train, kfold_X_train_features], kfold_y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks = [roc_auc_val])
    gc.collect()
    
    #model.load_weights(bst_model_path)
    model.load_weights(weights_path)
    
    predict += model.predict([X_test, test_features], batch_size=batch_size,verbose=1) / num_folds
    
    model.save_weights(best_weights_path + "model_fold" + str(cnt) + ".h5")
    gc.collect()
    # uncomment for out of fold predictions
    oof_predict[val_index] = model.predict([kfold_X_val, kfold_X_val_features], batch_size=batch_size, verbose=1)
    cv_score = roc_auc_score(kfold_y_val, oof_predict[val_index])
    
    scores.append(cv_score)
    cnt += 1

print("Done")
mean_score = np.mean(scores)
std_score = np.std(scores)
max_score = max(scores)
min_score = min(scores)
print('CV scores')
print('Mean: {0:.7f}'.format(mean_score))
print('Std: {0:.7f}'.format(std_score))
print('Min: {0:.7f}'.format(min_score))
print('Max: {0:.7f}'.format(max_score))

subm['injection'] = predict
subm_path = "submissions_validated/cnn_lstm_gru_attention_feat_elu_{}.csv".format(int(mean_score * 10**7))
subm.to_csv(subm_path, index=False)

runtime = time.time() - start_time
hours = runtime // 3600
minutes = (runtime - hours * 3600) // 60
print('Runtime: {}h {}min'.format(int(hours), int(minutes)))
print(runtime)