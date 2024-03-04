This RNN is the network without gain modulation and hebbian learning. It is more closed to what we want to realize compared with rnn_norm:
1. We gives random gains and shifts to the sigmoid function in the recurrent layer.
2. Training uses SGD on weights.
3. Dale's law is applied.

This is the RNN that used in norm situation. Codes are largely adapted from Ankit.
1. The activation function in the recurrent layer is the default sigmoid function, witout modifications in gains and shifts. 
2. Training uses Adam on weights.
3. No Dale's law.