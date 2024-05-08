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
                          "linear", "relu", "selu", "sigmoid", "softmax",
                          "softplus", "softsign", "swish", "tanh"]
    hyperarh = {
        "Layers": [1, 6],
    }

    for i in range(hyperarh["Layers"][1]):
        hyperarh["EncodingFilters"+str(i)] = [2, 64]
        hyperarh["EncodingActivation" + str(i)] = activations_choice
        hyperarh["DecodingFilters"+str(i)] = [2, 64]
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
        "upDropout": []}
    layers = hp.Int(name="Layers", min_value=hyperarh["Layers"][0],
                          max_value=hyperarh["Layers"][1])
    for i in range(layers):
        filters = hp.Int(name="EncodingFilters"+str(i), min_value=hyperarh["EncodingFilters"+str(i)][0],
                       max_value=hyperarh["EncodingFilters"+str(i)][1])
        activations = hp.Choice(name="EncodingActivation"+str(i), values=hyperarh["EncodingActivation"+str(i)])
        dropout = hp.Float(name="EncodingDropout"+str(i), min_value=0,
                       max_value=0.5)
        architecture["downFilters"].append(filters)
        architecture["downActivation"].append(activations)
        architecture["downDropout"].append(dropout)
        architecture["downMaxPool"].append(True)

        filters = hp.Int(name="DecodingFilters"+str(i), min_value=hyperarh["DecodingFilters"+str(i)][0],
                       max_value=hyperarh["DecodingFilters"+str(i)][1])
        activations = hp.Choice(name="DecodingActivation"+str(i), values=hyperarh["DecodingActivation"+str(i)])
        dropout = hp.Float(name="DecodingDropout"+str(i), min_value=0,
                       max_value=0.5)
        architecture["upFilters"].append(filters)
        architecture["upActivation"].append(activations)
        architecture["upDropout"].append(dropout)
    return architecture


class StockHyperModel(kt.HyperModel):
    def __init__(self, input_shape, loss, hyperarh=None):
        self.input_shape = input_shape
        self.loss = loss
        self.hyperarh = createdefaulthyperarhitecture() if hyperarh is None else hyperarh

    def build(self, hp):
        arhitecture = create_architecture_dictionary(hp, self.hyperarh)
        if os.environ.get('KERASTUNER_TUNER_ID', "chief") != "chief":
            for i in arhitecture.keys():
                print(i, arhitecture[i], end=", ")
            print()
        model = m.unet_model(self.input_shape, arhitecture)
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss=self.loss,
                      metrics=["Precision", "Recall", metrics.F1_Score()])
        return model


