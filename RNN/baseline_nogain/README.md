"norm" mean this RNN is quite closed to normal RNN. Codes are largely adapted from Ankit.
1. The activation function in the recurrent layer is the default sigmoid function, witout modifications in gains and shifts. 
2. Training uses Adam on weights.
3. No Dale's law.

"wt" means this RNN is the network without gain modulation and hebbian learning. It is more closed to what we want to realize:
1. We gives random gains and shifts to the sigmoid function in the recurrent layer.
2. Training uses SGD on weights.
3. Dale's law is applied.

"SIN" uses a input matrix, while "SIN2" uses a gaussian input receptive field.

"pt" means the RNN is trained for each input point, and it is trained on a continuous stream of input. Without "pt" means the RNN is trained for each epoch (300~400 input points).