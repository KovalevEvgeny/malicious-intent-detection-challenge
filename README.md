# malicious-intent-detection-challenge
https://www.kaggle.com/c/wallarm-ml-hackathon

### DAYS LEFT: 3

### SUBMISSIONS LEFT: 11 = 5*2 + 1 (today)

Metric: AUC-ROC

To do:

- ~~concatenate model with simple features (convert hardcoded rule to nice feature)~~
- ~~feature generation, TF-IDF~~
- make diverse models for stacking
- ~~more careful layers addition (think more what to add and how)~~
- hyperparameters optimization
- tune custom loss function to penalize for FP
- how the model actually predicts - maybe use weighted prediction
- ~~pick best fold to prediction~~
- preprocess texts (e.g. throw away \x08, ...)
- ~~drop duplicates~~
- ~~BatchNorm~~
- ensembling

*ran on kaggle kernel

|Attempt|Date|Mean CV | Std CV| Min CV| Max CV | Public | Place |Comment|Runtime|
|-|-|-|-|-|-|-|-|-|-|
|57*|09.12|0.9997821|**0.0000675**|**0.9996554**|0.9998705|0.99935|3/15|[CNN + LSTM + GRU + Simple Features (less) (reproduce)](https://www.kaggle.com/blackitten13/cnn-lstm-gru-addlessfeat3)|4:53|
|56|09.12|0.9997463|0.0001554|0.9994679|0.9999041|0.99981|3/15|[CNN + LSTM + GRU + Simple Features (less) on New Data (2)](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_addlessfeat_new2.py)|4:43|
|55|09.12|0.9997720|0.0000935|0.9995489|0.9998735|0.99977|2/15|[CNN + LSTM + GRU + Simple Features (less) on New Data](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_addlessfeat_new.py)|4:22|
|46|07.12|0.9997496|0.0001256|0.9995409|0.9998903|0.99919 (WTF)|2/15|[CNN + LSTM + GRU + Simple Features + BatchNorm](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_feat_bn.py)|5:01|
|58*|07.12|0.9994881|0.0001538|0.9992495|0.9997224|0.99975|-|[GRU + GRU](https://www.kaggle.com/blackitten13/gru2xx/notebook)|7:14|
|44|07.12|0.9997476|0.0001214|0.9994907|**0.9999382**|0.99977|2/15|[CNN + LSTM + GRU + Simple Features + Drop Duplicates](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_feat_dupl.py)|3:49|
|43|06.12|0.9995310|0.0001667|0.9992936|0.9998436|0.99957|2/15|[LSTM + GRU + CNN + Attention + Simple Features](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/LSTM_GRU_CNN_Attention_feat.py)|14:22|
|39|03.12|0.9998041|0.0001047|0.9995713|0.9999292|0.99984|1/15|[CNN + LSTM + GRU + Attention + Simple Features + ELU](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_Attention_feat_elu.py)|4:47|
|38|03.12|**0.9998239**|0.0000874|0.9996430|0.9999381|0.99984|1/15|[CNN + LSTM + GRU + Attention + Simple Features (less)](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_Attention_feat.py)|5:35|
|36|01.12|0.9997480|0.0001274|0.9994903|0.9999145|**0.99987**|1/14|[CNN + LSTM + GRU + Simple Features (less)](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_addlessfeat.py)|4:00|
|35|01.12|0.9998180|0.0001142|0.9995086|0.9999222|0.99984|1/14|[CNN + LSTM + GRU + Simple Features + decoded](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_addfeat_decoded.py)|5:12|
|34|01.12|0.9997933|0.0000790|0.9996502|0.9998798|0.99985|1/14|[CNN + LSTM + GRU + Simple Features](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_addfeat.py)|4:09|
|33|30.11|0.9997451|0.0001183|0.9995644|0.9999212|0.99980|1/12|[CNN + LSTM + GRU + BatchNorm](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_4.py)|4:47|
|31|29.11|0.9998148|0.0000936|0.9995960|0.9999355|0.99985|1/12|[CNN + LSTM + GRU Tuned 2](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_3.py)|4:56|
|29|28.11|0.9997988|0.0001029|0.9996361|0.9999263|0.99985|1/11|[CNN + LSTM + GRU Tuned](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_2.py)|5:36|
|28|27.11|0.9997558|?|?|?|0.99983|1/11|[CNN + LSTM + GRU + Attention](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_Attention.py)|3:06|
|27|27.11|0.9997032|?|?|0.9999318|0.99982|1/11|[LSTM + GRU + Attention (skip)](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/LSTM_GRU_Attention_skip.py)|5h|
|25|26.11|0.9997493|?|?|?|0.99985|  1/11 |[CNN + LSTM + GRU](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU.py)|?|
|23|26.11|0.9997200|?|?|?|0.99980|2/11|[CNN + GRU (5 folds)](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_GRU_full.py)|?|
