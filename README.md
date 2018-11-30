# malicious-intent-detection-challenge
https://www.kaggle.com/c/wallarm-ml-hackathon

Metric: AUC-ROC

To do:

- ~~concatenate model with simple features (convert hardcoded rule to nice feature)~~
- feature generation, TF-IDF
- make diverse models for stacking
- more careful layers addition (think more what to add and how)
- hyperparameters optimization
- tune custom loss function (AUC-ROC): https://stackoverflow.com/questions/43818584/custom-loss-function-in-keras


| Attempt | Mean CV | Std CV| Min CV| Max CV | Public | Place |Comment|Runtime|
|-|-|-|-|-|-|-|-|-|
|34|?|?|?|?|?|?|[CNN + LSTM + GRU + Simple Features](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_addfeat.py)|running...|
|33|0.9997451|0.0001183|0.9995644|0.9999212|0.99980|1/12|[CNN + LSTM + GRU + BatchNorm](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_4.py)|4:47|
|31|**0.9998148**|**0.0000936**|0.9995960|**0.9999355**|**0.99985**|1/12|[CNN + LSTM + GRU Tuned 2](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_3.py)|4:56|
|29|0.9997988|0.0001029|**0.9996361**|0.9999263|**0.99985**|1/11|[CNN + LSTM + GRU Tuned](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_2.py)|5:36|
|28|0.9997558|?|?|?|0.99983|1/11|[CNN + LSTM + GRU + Attention](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU_Attention.py)|3:06|
|27|0.9997032|?|?|0.9999318|0.99982|1/11|[LSTM + GRU + Attention (skip)](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/LSTM_GRU_Attention_skip.py)|5h|
|25 |0.9997493|?|?|?|**0.99985**|  1/11 |[CNN + LSTM + GRU](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU.py)|?|
|23|0.9997200|?|?|?|0.99980|2/11|[CNN + GRU (5 folds)](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_GRU_full.py)|?|
