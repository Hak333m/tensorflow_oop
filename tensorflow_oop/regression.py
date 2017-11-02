"""
Regression base models.
"""

from tensorflow_oop.neural_network import *
from tensorflow_oop.metrics import r2_score


class TFRegressor(TFNeuralNetwork):

    """
    Regression model with Mean squared loss function.

    Attributes:
        ...         Parrent class atributes.

    """

    def loss_function(self, targets, outputs):
        """Mean squared error.

        Arguments:
            targets     Tensor of batch with targets.
            outputs     Tensor of batch with outputs.

        Return:
            loss        Mean squared error operation.

        """
        return tf.losses.mean_squared_error(targets, outputs)

    @check_initialization
    @check_X_y_sample_weight
    def score(self, X, y, sample_weight=None):
        """Get coefficient of determination R^2 of the prediction.

        Arguments:
            X                  Batch of inputs values.
            y                  True labels for inputs values.
            sample_weight      Sample weights.

        Return:
            score              Coefficient of determination R^2.

        """
        return r2_score(y, self.predict(X), sample_weight=sample_weight, multioutput='variance_weighted')
