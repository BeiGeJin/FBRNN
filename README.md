/FNN: Feedforward Neural Network, mainly replicate the work of Swinehart and Abbott (2005)

    /training: the training scripts, since the model is simple it is written within each script. The main script is training_abb05_bphebb.py. Run it and it will save the trained model in /weights.

    /analysis: the analysis notebooks. The main one is analysis_abb05_bphebb.ipynb, containing the latest results and figures. Once you have the trained model, you can run this notebook to generate the figures.

    /perturbation: some experients on how the model will react to perturbations.

/RNN: Recurrent Neural Network

    /FORCE: the latest results where I use hebbian learning to modify the readout matrix. The main notebook is FORCE_tracking_full_p20.ipynb, containing the model and the simulation process. You should be able to run it without depending on other files. Another notebook worth looking at is FORCE_tracking_full_p20_adjout.ipynb, where I add a mechanism on adjusting the gain and the shift of the output neuron.

    /model: the RNN model classes for previous attmpts

    /ourRNN: previous attempts where I tried to use hebbian learning on the recurrent weight matrix but it does not work well.

/Overleaf: the weekly reports

/Control Theory Tests: some tests on toy examples of control theory


Note: since some organization efforts recently, there might be some path errors, especially in previous attempts. They are easy to fix and feel free to fix them.