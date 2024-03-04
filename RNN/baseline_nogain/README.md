This RNN is the network without gain modulation and hebbian learning. It is more closed to what we want to realize compared with rnn_norm:
1. We gives random gains and shifts to the sigmoid function in the recurrent layer.
2. Training uses SGD on weights.
3. Dale's law is applied.