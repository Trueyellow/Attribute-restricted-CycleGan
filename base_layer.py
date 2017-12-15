import keras.backend as K
from keras.utils import conv_utils
from keras.engine import Layer, InputSpec
from keras import initializers, regularizers, constraints



class ReflectPadding2D(Layer):
    """Reflection-padding layer for 2D input (e.g. picture).
    This layer adds rows and columns of reflected versions of the input at the
    top, bottom, left and right side of an image tensor.
    Parameters
    ----------
    padding : int, or tuple of 2 ints, or tuple of 2 tuples of 2 ints.
        If int, the same symmetric padding is applied to width and height. If
        tuple of 2 ints, interpreted as two different symmetric padding values
        for height and width: `(symmetric_height_pad, symmetric_width_pad)`. If
        tuple of 2 tuples of 2 ints: interpreted as `((top_pad, bottom_pad),
        (left_pad, right_pad))`
    data_format : str
        One of `channels_last` (default) or `channels_first`. The ordering of
        the dimensions in the inputs. `channels_last` corresponds to inputs
        with shape `(batch, height, width, channels)` while `channels_first`
        corresponds to inputs with shape `(batch, channels, height, width)`. It
        defaults to the `image_data_format` value found in your Keras config
        file at `~/.keras/keras.json`. If you never set it, then it will be
        "channels_last".
    Examples
    # --------
    # >>> import nethin.padding as padding
    # >>> from keras.layers import Input
    # >>> from keras.models import Model
    # >>> from keras import optimizers
    # >>> import numpy as np
    # >>>
    # >>> A = np.arange(12).reshape(3, 4).astype(np.float32)
    # >>>
    # >>> inputs = Input(shape=(3, 4, 1))
    # >>> x = neural.ReflectPadding2D(padding=2, data_format="channels_last")(inputs)
    # >>> model = Model(inputs=inputs, outputs=x)
    # >>> model.predict(A.reshape(1, 3, 4, 1)).reshape(7, 8)
    # array([[ 10.,   9.,   8.,   9.,  10.,  11.,  10.,   9.],
    #        [  6.,   5.,   4.,   5.,   6.,   7.,   6.,   5.],
    #        [  2.,   1.,   0.,   1.,   2.,   3.,   2.,   1.],
    #        [  6.,   5.,   4.,   5.,   6.,   7.,   6.,   5.],
    #        [ 10.,   9.,   8.,   9.,  10.,  11.,  10.,   9.],
    #        [  6.,   5.,   4.,   5.,   6.,   7.,   6.,   5.],
    #        [  2.,   1.,   0.,   1.,   2.,   3.,   2.,   1.]], dtype=float32)
    # >>>
    # >>> inputs = Input(shape=(1, 3, 4))
    # >>> x = neural.ReflectPadding2D(padding=1, data_format="channels_first")(inputs)
    # >>> model = Model(inputs=inputs, outputs=x)
    # >>> model.predict(A.reshape(1, 1, 3, 4)).reshape(5, 6)
    # array([[[[  5.,   4.,   5.,   6.,   7.,   6.],
    #          [  1.,   0.,   1.,   2.,   3.,   2.],
    #          [  5.,   4.,   5.,   6.,   7.,   6.],
    #          [  9.,   8.,   9.,  10.,  11.,  10.],
    #          [  5.,   4.,   5.,   6.,   7.,   6.]]]], dtype=float32)
    # """
    def __init__(self, padding=(1, 1), data_format=None, **kwargs):

        super(ReflectPadding2D, self).__init__(**kwargs)

        if isinstance(padding, int):
            self.padding = ((padding, padding), (padding, padding))
        elif hasattr(padding, '__len__'):
            if len(padding) != 2:
                raise ValueError('`padding` should have two elements. '
                                 'Found: ' + str(padding))
            height_padding = conv_utils.normalize_tuple(padding[0], 2,
                                                        "1st entry of padding")
            width_padding = conv_utils.normalize_tuple(padding[1], 2,
                                                       "2nd entry of padding")
            self.padding = (height_padding, width_padding)
        else:
            raise ValueError('`padding` should be either an int, '
                             'a tuple of 2 ints '
                             '(symmetric_height_pad, symmetric_width_pad), '
                             'or a tuple of 2 tuples of 2 ints '
                             '((top_pad, bottom_pad), (left_pad, right_pad)). '
                             'Found: ' + str(padding))

        self.data_format = "channels_last"

        self.input_spec = InputSpec(ndim=4)

    def build(self, input_shape):

        super(ReflectPadding2D, self).build(input_shape)

    def compute_output_shape(self, input_shape):

        if self.data_format == "channels_last":

            if input_shape[1] is not None:
                rows = input_shape[1] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None

            if input_shape[2] is not None:
                cols = input_shape[2] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None

            return (input_shape[0], rows, cols, input_shape[3])

        elif self.data_format == "channels_first":

            if input_shape[2] is not None:
                rows = input_shape[2] + self.padding[0][0] + self.padding[0][1]
            else:
                rows = None

            if input_shape[3] is not None:
                cols = input_shape[3] + self.padding[1][0] + self.padding[1][1]
            else:
                cols = None

            return (input_shape[0], input_shape[1], rows, cols)

    def get_config(self):

        config = {"padding": self.padding,
                  "data_format": self.data_format}
        base_config = super(ReflectPadding2D, self).get_config()

        return dict(list(base_config.items()) + list(config.items()))

    def call(self, inputs):
        """Performs the actual padding.
        Parameters
        ----------
        inputs : Tensor, rank 4
            4D tensor with shape:
                - If `data_format` is `"channels_last"`:
                    `(batch, rows, cols, channels)`
                - If `data_format` is `"channels_first"`:
                    `(batch, channels, rows, cols)`
        Returns
        -------
        outputs : Tensor, rank 4
            4D tensor with shape:
                - If `data_format` is `"channels_last"`:
                    `(batch, padded_rows, padded_cols, channels)`
                - If `data_format` is `"channels_first"`:
                    `(batch, channels, padded_rows, padded_cols)`
        """
        outputs = K.spatial_2d_padding(inputs,
                                       padding=self.padding,
                                       data_format=self.data_format)

        p00, p01 = self.padding[0][0], self.padding[0][1]
        p10, p11 = self.padding[1][0], self.padding[1][1]
        if self.data_format == "channels_last":

            row0 = K.concatenate([inputs[:, p00:0:-1, p10:0:-1, :],
                                  inputs[:, p00:0:-1, :, :],
                                  inputs[:, p00:0:-1, -2:-2-p11:-1, :]],
                                 axis=2)
            row1 = K.concatenate([inputs[:, :, p10:0:-1, :],
                                  inputs,
                                  inputs[:, :, -2:-2-p11:-1, :]],
                                 axis=2)
            row2 = K.concatenate([inputs[:, -2:-2-p01:-1, p10:0:-1, :],
                                  inputs[:, -2:-2-p01:-1, :, :],
                                  inputs[:, -2:-2-p01:-1, -2:-2-p11:-1, :]],
                                 axis=2)

            outputs = K.concatenate([row0, row1, row2], axis=1)

        else:  # self.data_format == "channels_first"

            row0 = K.concatenate([inputs[:, :, p00:0:-1, p10:0:-1],
                                  inputs[:, :, p00:0:-1, :],
                                  inputs[:, :, p00:0:-1, -2:-2-p11:-1]],
                                 axis=3)
            row1 = K.concatenate([inputs[:, :, :, p10:0:-1],
                                  inputs,
                                  inputs[:, :, :, -2:-2-p11:-1]],
                                 axis=3)
            row2 = K.concatenate([inputs[:, :, -2:-2-p01:-1, p10:0:-1],
                                  inputs[:, :, -2:-2-p01:-1, :],
                                  inputs[:, :, -2:-2-p01:-1, -2:-2-p11:-1]],
                                 axis=3)

            outputs = K.concatenate([row0, row1, row2], axis=2)

        return outputs


class InstanceNormalization(Layer):
    """Instance normalization layer (Lei Ba et al, 2016, Ulyanov et al., 2016).
    Normalize the activations of the previous layer at each step,
    i.e. applies a transformation that maintains the mean activation
    close to 0 and the activation standard deviation close to 1.
    # Arguments
        axis: Integer, the axis that should be normalized
            (typically the features axis).
            For instance, after a `Conv2D` layer with
            `data_format="channels_first"`,
            set `axis=1` in `InstanceNormalization`.
            Setting `axis=None` will normalize all values in each instance of the batch.
            Axis 0 is the batch dimension. `axis` cannot be set to 0 to avoid errors.
        epsilon: Small float added to variance to avoid dividing by zero.
        center: If True, add offset of `beta` to normalized tensor.
            If False, `beta` is ignored.
        scale: If True, multiply by `gamma`.
            If False, `gamma` is not used.
            When the next layer is linear (also e.g. `nn.relu`),
            this can be disabled since the scaling
            will be done by the next layer.
        beta_initializer: Initializer for the beta weight.
        gamma_initializer: Initializer for the gamma weight.
        beta_regularizer: Optional regularizer for the beta weight.
        gamma_regularizer: Optional regularizer for the gamma weight.
        beta_constraint: Optional constraint for the beta weight.
        gamma_constraint: Optional constraint for the gamma weight.
    # Input shape
        Arbitrary. Use the keyword argument `input_shape`
        (tuple of integers, does not include the samples axis)
        when using this layer as the first layer in a model.
    # Output shape
        Same shape as input.
    # References
        - [Layer Normalization](https://arxiv.org/abs/1607.06450)
        - [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
    """
    def __init__(self,
                 axis=None,
                 epsilon=1e-3,
                 center=True,
                 scale=True,
                 beta_initializer='zeros',
                 gamma_initializer='ones',
                 beta_regularizer=None,
                 gamma_regularizer=None,
                 beta_constraint=None,
                 gamma_constraint=None,
                 **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.supports_masking = True
        self.axis = axis
        self.epsilon = epsilon
        self.center = center
        self.scale = scale
        self.beta_initializer = initializers.get(beta_initializer)
        self.gamma_initializer = initializers.get(gamma_initializer)
        self.beta_regularizer = regularizers.get(beta_regularizer)
        self.gamma_regularizer = regularizers.get(gamma_regularizer)
        self.beta_constraint = constraints.get(beta_constraint)
        self.gamma_constraint = constraints.get(gamma_constraint)

    def build(self, input_shape):
        ndim = len(input_shape)
        if self.axis == 0:
            raise ValueError('Axis cannot be zero')

        if (self.axis is not None) and (ndim == 2):
            raise ValueError('Cannot specify axis for rank 1 tensor')

        self.input_spec = InputSpec(ndim=ndim)

        if self.axis is None:
            shape = (1,)
        else:
            shape = (input_shape[self.axis],)

        if self.scale:
            self.gamma = self.add_weight(shape=shape,
                                         name='gamma',
                                         initializer=self.gamma_initializer,
                                         regularizer=self.gamma_regularizer,
                                         constraint=self.gamma_constraint)
        else:
            self.gamma = None
        if self.center:
            self.beta = self.add_weight(shape=shape,
                                        name='beta',
                                        initializer=self.beta_initializer,
                                        regularizer=self.beta_regularizer,
                                        constraint=self.beta_constraint)
        else:
            self.beta = None
        self.built = True

    def call(self, inputs, training=None):
        input_shape = K.int_shape(inputs)
        reduction_axes = list(range(0, len(input_shape)))

        if self.axis is not None:
            del reduction_axes[self.axis]

        del reduction_axes[0]

        mean = K.mean(inputs, reduction_axes, keepdims=True)
        stddev = K.std(inputs, reduction_axes, keepdims=True) + self.epsilon
        normed = (inputs - mean) / stddev

        broadcast_shape = [1] * len(input_shape)
        if self.axis is not None:
            broadcast_shape[self.axis] = input_shape[self.axis]

        if self.scale:
            broadcast_gamma = K.reshape(self.gamma, broadcast_shape)
            normed = normed * broadcast_gamma
        if self.center:
            broadcast_beta = K.reshape(self.beta, broadcast_shape)
            normed = normed + broadcast_beta
        return normed

    def get_config(self):
        config = {
            'axis': self.axis,
            'epsilon': self.epsilon,
            'center': self.center,
            'scale': self.scale,
            'beta_initializer': initializers.serialize(self.beta_initializer),
            'gamma_initializer': initializers.serialize(self.gamma_initializer),
            'beta_regularizer': regularizers.serialize(self.beta_regularizer),
            'gamma_regularizer': regularizers.serialize(self.gamma_regularizer),
            'beta_constraint': constraints.serialize(self.beta_constraint),
            'gamma_constraint': constraints.serialize(self.gamma_constraint)
        }
        base_config = super(InstanceNormalization, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


