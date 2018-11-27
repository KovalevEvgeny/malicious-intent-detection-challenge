import time
start_time = time.time()
import warnings
warnings.filterwarnings('ignore')

import msgpack
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(13)
import tensorflow as tf
tf.set_random_seed(seed=13)
from keras import optimizers, regularizers
from keras.engine.topology import Layer
from keras.models import Model
from keras.layers import MaxPooling1D, Dense, Input, Embedding, Dropout, CuDNNGRU, CuDNNLSTM, Conv1D, Bidirectional
from keras.preprocessing import text as keras_text, sequence as keras_seq
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc
from keras import backend as K

from utils import apply_rules

import os
os.environ['OMP_NUM_THREADS'] = '7'


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


weights_path = "weights_validation/cnn_lstm_gru_attention.h5"


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
                print("*** New High Score (previous: %.6f) \n" % self.max_score)
                model.save_weights(weights_path)
                self.max_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 3:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True


CONTEXT_DIM = 100
class Attention(Layer):

    def __init__(self, regularizer=regularizers.l2(1e-10), **kwargs):
        self.regularizer = regularizer
        self.supports_masking = True
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3        
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[-1], CONTEXT_DIM),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)
        self.b = self.add_weight(name='b',
                                 shape=(CONTEXT_DIM,),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)
        self.u = self.add_weight(name='u',
                                 shape=(CONTEXT_DIM,),
                                 initializer='normal',
                                 trainable=True, 
                                 regularizer=self.regularizer)        
        super(Attention, self).build(input_shape)

    @staticmethod
    def softmax(x, dim):
        """Computes softmax along a specified dim. Keras currently lacks this feature.
        """
        if K.backend() == 'tensorflow':
            import tensorflow as tf
            return tf.nn.softmax(x, dim)
        elif K.backend() == 'theano':
            # Theano cannot softmax along an arbitrary dim.
            # So, we will shuffle `dim` to -1 and un-shuffle after softmax.
            perm = np.arange(K.ndim(x))
            perm[dim], perm[-1] = perm[-1], perm[dim]
            x_perm = K.permute_dimensions(x, perm)
            output = K.softmax(x_perm)

            # Permute back
            perm[dim], perm[-1] = perm[-1], perm[dim]
            output = K.permute_dimensions(x, output)
            return output
        else:
            raise ValueError("Backend '{}' not supported".format(K.backend()))

    def call(self, x, mask=None):
        ut = K.tanh(K.bias_add(K.dot(x, self.W), self.b)) * self.u

        # Collapse `attention_dims` to 1. This indicates the weight for each time_step.
        ut = K.sum(ut, axis=-1, keepdims=True)

        # Convert those weights into a distribution but along time axis.
        # i.e., sum of alphas along `time_steps` axis should be 1.
        self.at = self.softmax(ut, dim=1)
        if mask is not None:
            self.at *= K.cast(K.expand_dims(mask, -1), K.floatx())

        # Weighted sum along `time_steps` axis.
        return K.sum(x * self.at, axis=-2)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])
    
    def get_config(self):
        config = {}
        base_config = super(Attention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_mask(self, inputs, mask):
        return None


def build_model(embed_size=512, dropout=0.1, clipvalue=1.0):
    inp = Input(shape=(maxlen, ))
    x = Embedding(len(tokenizer.word_counts) + 1, embed_size)(inp)
    x = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(CuDNNLSTM(60, return_sequences=True))(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(CuDNNGRU(60, return_sequences=True))(x)
    x = Dropout(dropout)(x)
    x = Attention()(x)
    x = Dropout(dropout)(x)
    x = Dense(50, activation='relu')(x)
    x = Dropout(dropout)(x)
    x = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=inp, outputs=x)
    adam = optimizers.adam(clipvalue=clipvalue)
    model.compile(loss='binary_crossentropy',
                  optimizer=adam,
                  metrics=['accuracy'])
    return model


model = build_model()

batch_size = 32
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
    kfold_X_val = X_train[val_index]
    
    gc.collect()
    K.clear_session()
    
    model = build_model()
    
    roc_auc_val = RocAucEvaluation(validation_data=(kfold_X_val, kfold_y_val))
    
    model.fit(kfold_X_train, kfold_y_train, batch_size=batch_size, epochs=epochs, verbose=1, callbacks = [roc_auc_val])
    gc.collect()
    
    #model.load_weights(bst_model_path)
    model.load_weights(weights_path)
    
    predict += model.predict(X_test, batch_size=batch_size,verbose=1) / num_folds
    
    gc.collect()
    # uncomment for out of fold predictions
    oof_predict[val_index] = model.predict(kfold_X_val, batch_size=batch_size, verbose=1)
    cv_score = roc_auc_score(kfold_y_val, oof_predict[val_index])
    
    scores.append(cv_score)
    print('score: ',cv_score)
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
subm_path = "submissions_validated/cnn_lstm_gru_attention_{}.csv".format(int(mean_score * 10**7))
subm.to_csv(subm_path, index=False)

apply_rules(test, subm_path, output_path='submissions_rules/cnn_lstm_gru_attention_rules.csv')

runtime = time.time() - start_time
hours = runtime // 3600
minutes = runtime - hours * 3600
print('Runtime: {}h {}min'.format(int(hours), int(minutes)))
print(runtime)