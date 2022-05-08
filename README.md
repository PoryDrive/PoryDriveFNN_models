# PoryDriveFNN_models
PoryDriveFNN trained models from 1GB and 2GB datasets.

All these are trained with the default Tensorflow Keras weight initialiser [GlorotNormal](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/GlorotNormal).

#### notes
- `tanh` and `selu` seem to perform best.
- `softsign` and `relu` seems to have potential.

In [train.py](https://github.com/PoryDrive/PoryDriveFNN/blob/main/train.py) I went for a large amount of units per layer with only a few layers, 4 max. This didn't seem to scale well as the performance of larger networks did seem to get a little better in certain conditions but only marginally so.

In [train2.py](https://github.com/PoryDrive/PoryDriveFNN/blob/main/train2.py) I went for the opposite, more layers but fewer units per layer. This seems to scale much better, smaller networks seem to perform better than larger networks from `train.py` using both `tanh` and `selu` activation functions.

`ReLU` shows potential in `train2.py` networks but it's lacking something, the networks tend to become stalkers and not "collectors" which is a rare but notable outcome that can occur, a network will sometimes train to only follow the porygon and never _(extremely rarily)_ make contact with it.

`SoftSign` had a collection rate of roughly half that of `tanh` and `selu`, still I think it could be worth investigation.

#### noteable models
- [nesterov_16_32_32_shuf](SELU2_tested_from_1gb_dataset/HIGH/nesterov_16_32_32_shuf/)
- [tanh_nesterov_16_32_32_shuf](Various_tested_from_1gb_dataset/HIGH/tanh_nesterov_16_32_32_shuf/)
