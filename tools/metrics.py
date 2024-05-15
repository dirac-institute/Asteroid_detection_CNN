import tensorflow as tf
import numpy as np
def SumLosses(losses):
    """
    Sum of multiple losses, passed to model as loss function.
    :param losses: list of loss functions
    :return: sum of the losses
    """
    def custom_loss(y_true, y_pred):
        loss = 0
        for i, l in enumerate(losses):
            loss += l(y_true, y_pred)
        return loss
    return custom_loss

def tversky(y_true, y_pred, alpha=0.9):
    """
    Twersky coefficient, passed to model as metric.
    :param y_true: Targets for training
    :param y_pred: Predictions from model
    :param alpha: False negaive control parameter, if 0.5 twersky coefficient becomes dice coefficient
    :return: Tversky coefficient
    """
    smooth = 1
    y_true_pos = tf.reshape(y_true, [-1])
    y_pred_pos = tf.reshape(y_pred, [-1])
    true_pos = tf.reduce_sum(y_true_pos * y_pred_pos, axis=None)
    false_neg = tf.reduce_sum(y_true_pos * (1-y_pred_pos), axis=None)
    false_pos = tf.reduce_sum((1-y_true_pos)*y_pred_pos, axis=None)
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    """
    Twersky loss, passed to model as loss function.
    :param y_true: Targets for training
    :param y_pred: Predictions from model
    :return: Tversky loss
    """
    return 1 - tversky(y_true, y_pred)

def FocalTversky (alpha=0.9, gamma=2):
    """
    Focal Tversky loss, parses the function for use as a loss function.
    :param alpha: False negative control parameter, if 0.5 twersky coefficient becomes dice coefficient
    :param gamma: Focal parameter for loss function (default 2)
    :return: loss function
    """
    def focal_tversky(y_true, y_pred):
        pt_1 = tversky(y_true, y_pred, alpha)
        return tf.math.pow((1 - pt_1), (gamma))
    return focal_tversky


class F1_Score(tf.keras.metrics.Metric):
    """
    Pixel-wise F1 score metric, passed to model as metric.
    """
    def __init__(self, name='f1_score', **kwargs):
        super().__init__(name=name, **kwargs)
        self.f1 = self.add_weight(name='f1', initializer='zeros')
        self.counter = self.add_weight(name='counter', initializer='zeros')
        self.precision_fn = tf.keras.metrics.Precision(thresholds=0.5)
        self.recall_fn = tf.keras.metrics.Recall(thresholds=0.5)
        self.count = self.add_weight(name='F1ScoreCount', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        p = self.precision_fn(y_true, y_pred)
        r = self.recall_fn(y_true, y_pred)
        self.f1.assign_add(2 * ((p * r) / (p + r + 1e-6)))
        self.count.assign_add(1)


    def result(self):
        return self.f1/self.count

    def reset_state(self):
        # we also need to reset the state of the precision and recall objects
        self.precision_fn.reset_state()
        self.recall_fn.reset_state()
        self.f1.assign(0)
        self.count.assign(0)