import msgpack
import pandas as pd


def apply_rules(test, subm_path, output_path):
    submission = pd.read_csv(subm_path)
    test['length'] = test['text'].apply(lambda x: len(x))
    long_text_idx = test[test['length'] > 1000].index
    submission.loc[long_text_idx]['injection'] = 1.0
    submission.to_csv(output_path, index=False)
    

with open('test.msgpack', 'rb') as data_file:
    test = msgpack.unpack(data_file)
test = pd.DataFrame(test)
test.columns = ['id', 'text']

subm_path = 'submissions_validated/cnn_gru_validated_9997200.csv'
output_path = 'submissions_rules/cnn_gru_validated_rules.csv'

apply_rules(test, subm_path, output_path)