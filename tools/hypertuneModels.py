if __name__ == "__main__":
    import model as m
    import metrics
else:
    import tools.model as m
    import tools.metrics as metrics
import keras_tuner as kt
import numpy as np
import tensorflow as tf
import os

def get_best_hyperparameters(tuner, num_trials=1):
    trials = [t
              for t in tuner.oracle.trials.values()
              if (t.status == kt.engine.trial.TrialStatus.COMPLETED) & (not np.isnan(t.score))]
    sorted_trials = sorted(trials, key=lambda trial: trial.score, reverse=tuner.oracle.objective.direction == "max")
    hyperparameters = [trial.hyperparameters for trial in sorted_trials if not np.isnan(trial.score)]
    return hyperparameters[:num_trials]

def createdefaulthyperarhitecture():
    # do not use "exponential" as it suffers from vanishing gradient problem
    activations_choice = ["elu", "gelu", "hard_sigmoid",
                          "linear", "relu", "selu", "sigmoid",
                          "softplus", "softsign", "swish", "tanh"]
    hyperarh = {
        "Layers": [3, 7],
    }

    for i in range(hyperarh["Layers"][1]):
        hyperarh["EncodingFilters"+str(i)] = [2**i, 2**(i+1)]
        hyperarh["EncodingActivation" + str(i)] = activations_choice
        hyperarh["DecodingFilters"+str(i)] = [2**i, 2**(i+1)]
        hyperarh["DecodingActivation" + str(i)] = activations_choice
    return hyperarh

def create_architecture_dictionary(hp, hyperarh):
    architecture = {
        "downFilters": [],
        "downActivation": [],
        "downDropout": [],
        "downMaxPool": [],
        "upFilters": [],
        "upActivation": [],
        "upDropout": [],
        "attention": []}
    up_layers = hp.Int(name="UpLayers", min_value=hyperarh["Layers"][0],
                          max_value=hyperarh["Layers"][1])

    down_layers = up_layers-2
    for i in range(up_layers):
        filters = hp.Int(name="EncodingFilters"+str(i), min_value=hyperarh["EncodingFilters"+str(i)][0],
                       max_value=hyperarh["EncodingFilters"+str(i)][1])
        activations = hp.Choice(name="EncodingActivation"+str(i), values=hyperarh["EncodingActivation"+str(i)])
        dropout = hp.Float(name="EncodingDropout"+str(i), min_value=0,
                       max_value=0.5)
        architecture["downFilters"].append(filters)
        architecture["downActivation"].append(activations)
        architecture["downDropout"].append(dropout)
        architecture["downMaxPool"].append(True)
        architecture["attention"].append(True)

    for i in range(down_layers):
        filters = hp.Int(name="DecodingFilters"+str(i), min_value=hyperarh["DecodingFilters"+str(i)][0],
                       max_value=hyperarh["DecodingFilters"+str(i)][1])
        activations = hp.Choice(name="DecodingActivation"+str(i), values=hyperarh["DecodingActivation"+str(i)])
        dropout = hp.Float(name="DecodingDropout"+str(i), min_value=0, max_value=0.5)
        architecture["upFilters"].append(filters)
        architecture["upActivation"].append(activations)
        architecture["upDropout"].append(dropout)
    return architecture


def create_training_dictionary(hp):
    training_dict = {
        "alpha": hp.Float(name="Alpha", min_value=0.4, max_value=(1-1e-7)),
        "gamma": hp.Float(name="Gamma", min_value=0.5, max_value=10),
        "LR": hp.Float(name="LearningRate", min_value=1e-7, max_value=0.1)}
    return training_dict


class StockHyperModel(kt.HyperModel):
    def __init__(self, input_shape, arhitecture=None, training_parameters=None, hyperarh=None):
        self.input_shape = input_shape
        self.arhitecture = arhitecture
        self.training_parameters = training_parameters
        self.hyperarh = createdefaulthyperarhitecture() if hyperarh is None else hyperarh

    def build(self, hp):
        if self.arhitecture is None:
            self.arhitecture = create_architecture_dictionary(hp, self.hyperarh)
        if self.training_parameters is None:
            self.training_parameters = create_training_dictionary(hp)
        loss = tools.metrics.FocalTversky(alpha=self.training_parameters["alpha"],
                                          gamma=self.training_parameters["gamma"])
        if os.environ.get('KERASTUNER_TUNER_ID', "chief") != "chief":
            for i in arhitecture.keys():
                print(i, arhitecture[i], end=", ")
            print()
        model = m.unet_model(self.input_shape, self.arhitecture)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.training_parameters["LR"]), loss=loss,
                      metrics=["Precision", "Recall", metrics.F1_Score()])
        return model


