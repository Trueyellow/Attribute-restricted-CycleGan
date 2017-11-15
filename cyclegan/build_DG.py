import keras.backend as K
from keras.layers import Activation, Input, Dropout
from keras.layers import LeakyReLU, BatchNormalization
from keras.models import Model
from keras.layers import Conv2D, UpSampling2D
from keras.layers.merge import Add, concatenate
from cyclegan.base_layer import InstanceNormalization, ReflectPadding2D

padding = ReflectPadding2D


def normalize():
    #   return BatchNormalization(axis=get_filter_dim())
    return InstanceNormalization()


def scaleup(input, ngf, kss, strides, padding):
    #   x = Conv2DTranspose(ngf, kss, strides=strides, padding=padding)(input)

    # upsample + conv

    x = UpSampling2D(strides)(input)
    x = Conv2D(ngf, kss, padding=padding)(x)
    return x

# for building resnet_6 block
def res_block(input, filters, kernel_size=(3, 3), strides=(1, 1)):
    # conv_block:add(nn.SpatialReflectionPadding(1, 1, 1, 1))
    # conv_block:add(nn.SpatialConvolution(dim, dim, 3, 3, 1, 1, p, p))
    # conv_block:add(normalization(dim))
    # conv_block:add(nn.ReLU(true))
    x = padding()(input)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides, )(x)
    x = normalize()(x)
    x = Activation('relu')(x)

    x = padding()(x)
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               strides=strides, )(x)
    x = normalize()(x)

    #   merged = Concatenate(axis=get_filter_dim())([input, x])
    merged = Add()([input, x])
    return merged


# basic discriminator
def basic_D(input_shape, ndf, n_layers=3, kw=4, dropout=0.0, use_sigmoid=False, **kwargs):

    input = Input(input_shape)
    x = Conv2D(ndf, (kw, kw), strides=(2, 2), padding='same')(input)
    x = LeakyReLU(0.2)(x)

    for i in range(n_layers - 1):
        x = Conv2D(ndf * min(2 ** (i + 1), 8), (kw, kw), strides=(2, 2), padding='same')(x)
        x = InstanceNormalization()(x)
        if dropout > 0.: x = Dropout(dropout)(x)
        x = LeakyReLU(0.2)(x)

    x = Conv2D(ndf * min(2 ** (n_layers + 1), 8), (kw, kw), strides=(1, 1), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, (kw, kw), strides=(1, 1), padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    model = Model(input, x, name=kwargs.get('name', None))
    print('Model basic D:')
    model.summary()
    return model


# basic resnet 6 blocks generator
def resnet_6blocks(input_shape, output_nc, ngf, **kwargs):
    ks = 3
    f = 7
    p = int((f - 1) / 2)

    input = Input(input_shape)
    # local e1 = data - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(3, ngf, f, f, 1, 1) - normalization(ngf) - nn.ReLU(true)
    x = padding(p)(input)
    x = Conv2D(ngf, (f, f))(x)
    x = normalize()(x)
    x = Activation('relu')(x)

    # local e2 = e1 - nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
    x = Conv2D(ngf * 2, (ks, ks), strides=(2, 2), padding='same')(x)
    x = normalize()(x)
    x = Activation('relu')(x)

    # local e3 = e2 - nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1) - normalization(ngf*4) - nn.ReLU(true)
    x = Conv2D(ngf * 4, (ks, ks), strides=(2, 2), padding='same')(x)
    x = normalize()(x)
    x = Activation('relu')(x)

    # local d1 = e3 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
    #  - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)

    # local d2 = d1 - nn.SpatialFullConvolution(ngf*4, ngf*2, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf*2) - nn.ReLU(true)
    # x = Conv2DTranspose(ngf*2, (ks,ks), strides=(2,2), padding='same')(x)
    x = scaleup(x, ngf * 2, (ks, ks), strides=(2, 2), padding='same')
    x = normalize()(x)
    x = Activation('relu')(x)

    # local d3 = d2 - nn.SpatialFullConvolution(ngf*2, ngf, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf) - nn.ReLU(true)
    # x = Conv2DTranspose(ngf, (ks,ks), strides=(2,2), padding='same')(x)
    x = scaleup(x, ngf, (ks, ks), strides=(2, 2), padding='same')
    x = normalize()(x)
    x = Activation('relu')(x)

    # local d4 = d3 - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1) - nn.Tanh()
    x = padding(p)(x)
    x = Conv2D(output_nc, (f, f))(x)
    x = Activation('tanh')(x)

    model = Model(input, x, name=kwargs.get('name', None))
    print('Model resnet 6blocks:')
    model.summary()
    return model

# basic resnet 6 blocks generator for A(with label)
def resnet_6blocks_A(input_shape, label_shape, output_nc, ngf, **kwargs):
    ks = 3
    f = 7
    p = int((f - 1) / 2)

    input = Input(input_shape)
    label = Input(label_shape)
    # local e1 = data - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(3, ngf, f, f, 1, 1) - normalization(ngf) - nn.ReLU(true)
    x = padding(p)(input)
    x = Conv2D(ngf, (f, f))(x)
    x_conca = concatenate([x, label])
    x = normalize()(x_conca)
    x = Activation('relu')(x)

    # local e2 = e1 - nn.SpatialConvolution(ngf, ngf*2, ks, ks, 2, 2, 1, 1) - normalization(ngf*2) - nn.ReLU(true)
    x = Conv2D(ngf * 2, (ks, ks), strides=(2, 2), padding='same')(x)
    x = normalize()(x)
    x = Activation('relu')(x)

    # local e3 = e2 - nn.SpatialConvolution(ngf*2, ngf*4, ks, ks, 2, 2, 1, 1) - normalization(ngf*4) - nn.ReLU(true)
    x = Conv2D(ngf * 4, (ks, ks), strides=(2, 2), padding='same')(x)
    x = normalize()(x)
    x = Activation('relu')(x)

    # local d1 = e3 - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
    #  - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type) - build_res_block(ngf*4, padding_type)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)
    x = res_block(x, ngf * 4)

    # local d2 = d1 - nn.SpatialFullConvolution(ngf*4, ngf*2, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf*2) - nn.ReLU(true)
    # x = Conv2DTranspose(ngf*2, (ks,ks), strides=(2,2), padding='same')(x)
    x = scaleup(x, ngf * 2, (ks, ks), strides=(2, 2), padding='same')
    x = normalize()(x)
    x = Activation('relu')(x)

    # local d3 = d2 - nn.SpatialFullConvolution(ngf*2, ngf, ks, ks, 2, 2, 1, 1,1,1) - normalization(ngf) - nn.ReLU(true)
    # x = Conv2DTranspose(ngf, (ks,ks), strides=(2,2), padding='same')(x)
    x = scaleup(x, ngf, (ks, ks), strides=(2, 2), padding='same')
    x = normalize()(x)
    x = Activation('relu')(x)

    # local d4 = d3 - nn.SpatialReflectionPadding(p, p, p, p) - nn.SpatialConvolution(ngf, output_nc, f, f, 1, 1) - nn.Tanh()
    x = padding(p)(x)
    x = Conv2D(output_nc, (f, f))(x)
    x = Activation('tanh')(x)

    model = Model([input, label], x, name=kwargs.get('name', None))
    print('Model resnet 6blocks:')
    model.summary()
    return model

# basic discriminator for A(with label)
def basic_D_A(input_shape, label_shape, ndf, n_layers=3, kw=4, dropout=0.0, use_sigmoid=False, **kwargs):

    input = Input(input_shape)
    label = Input(label_shape)
    x = Conv2D(ndf, (kw, kw), strides=(2, 2), padding='same')(input)
    x = concatenate([x, label])
    x = LeakyReLU(0.2)(x)

    for i in range(n_layers - 1):
        x = Conv2D(ndf * min(2 ** (i + 1), 8), (kw, kw), strides=(2, 2), padding='same')(x)
        x = InstanceNormalization()(x)
        if dropout > 0.: x = Dropout(dropout)(x)
        x = LeakyReLU(0.2)(x)

    x = Conv2D(ndf * min(2 ** (n_layers + 1), 8), (kw, kw), strides=(1, 1), padding='same')(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)

    x = Conv2D(1, (kw, kw), strides=(1, 1), padding='same')(x)
    if use_sigmoid:
        x = Activation('sigmoid')(x)

    model = Model([input, label], x, name=kwargs.get('name', None))
    print('Model basic D:')
    model.summary()
    return model

