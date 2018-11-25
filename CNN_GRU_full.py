import warnings
warnings.filterwarnings('ignore')

import msgpack
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
np.random.seed(13)
import tensorflow as tf
tf.set_random_seed(seed=13)
from keras import optimizers
from keras.models import Model
from keras.layers import MaxPooling1D, GlobalMaxPool1D, Dense, Input, Embedding, Dropout, CuDNNGRU, Conv1D, Bidirectional
from keras.preprocessing import text as keras_text, sequence as keras_seq
from keras.callbacks import Callback
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
import gc
from keras import backend as K

import os
os.environ['OMP_NUM_THREADS'] = '4'


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
                model.save_weights("weights_validation/cnn_gru_best_weights.h5")
                self.max_score=score
                self.not_better_count = 0
            else:
                self.not_better_count += 1
                if self.not_better_count > 3:
                    print("Epoch %05d: early stopping, high score = %.6f" % (epoch,self.max_score))
                    self.model.stop_training = True


def build_model(embed_size=512, dropout=0.1, clipvalue=1.0):
    inp = Input(shape=(maxlen, ))
    x = Embedding(len(tokenizer.word_counts) + 1, embed_size)(inp)
    x = Conv1D(filters=100, kernel_size=4, padding='same', activation='relu')(x)
    x = MaxPooling1D(pool_size=4)(x)
    x = Dropout(dropout)(x)
    x = Bidirectional(CuDNNGRU(60, return_sequences=True))(x)
    x = GlobalMaxPool1D()(x)
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

num_folds = 5

predict = np.zeros((test.shape[0],1))

scores = []
oof_predict = np.zeros((train.shape[0],1))

kf = KFold(n_splits=num_folds, shuffle=True, random_state=13)

for train_index, val_index in kf.split(X_train):
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
    model.load_weights("weights_validation/cnn_gru_best_weights.h5")
    
    predict += model.predict(X_test, batch_size=batch_size,verbose=1) / num_folds
    
    gc.collect()
    # uncomment for out of fold predictions
    oof_predict[val_index] = model.predict(kfold_X_val, batch_size=batch_size, verbose=1)
    cv_score = roc_auc_score(kfold_y_val, oof_predict[val_index])
    
    scores.append(cv_score)
    print('score: ',cv_score)

print("Done")
cv_score = np.mean(scores)
print('Total CV score is {}'.format(cv_score))

subm['injection'] = predict
subm.to_csv("submissions_validated/cnn_gru_validated_{}.csv".format(int(cv_score * 10**7)), index=False)