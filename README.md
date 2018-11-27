# malicious-intent-detection-challenge
https://www.kaggle.com/c/wallarm-ml-hackathon

To do:

- concatenate model with simple features (convert hardcoded rule to nice feature)
- feature generation, TF-IDF
- make diverse models for stacking
- more careful layers addition (think more what to add and how)
- hyperparameters optimization


| Attempt | Mean CV | Std CV| Min CV| Max CV | Public | Place |Comment|Folds|Runtime|
|-|-|-|-|-|-|-|-|-|-|
|?|?|?|?|?|?|?|[CNN + LSTM + GRU + Attention]|10|running...|
|27|0.9997032|?|?|**0.9999318**|0.99982|1/11|[LSTM + GRU + Attention (skip)](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/LSTM_GRU_Attention_skip.py)|10|5h|
|25 | **0.9997493**|?|?|?|**0.99985**|  1/11 |[CNN + LSTM + GRU](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_LSTM_GRU.py)|10|?|
|23|0.9997200|?|?|?|0.99980|2/11|[CNN + GRU](https://github.com/blacKitten13/malicious-intent-detection-challenge/blob/master/CNN_GRU_full.py)|5|?|
