import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Reshape, Dropout, Conv1D, Conv2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Flatten

def SeqCNNEnc(input_shape, latent_dim, n_filters, k_size, dropout=0.0):
    inputs = Input(shape=input_shape)
    x = inputs
    for f in n_filters:
        x = Conv1D(f, k_size, padding='same', strides=2, activation='relu')(x)
        x = Dropout(dropout)(x)
        # x = MaxPooling1D(2)(x)
    x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    outputs = Dense(latent_dim)(x)
    return Model(inputs, outputs, name='seq_encoder')

def SeqCNNDec(input_shape, ts_shape, n_filters, k_size):
    inputs = Input(shape=input_shape)
    x = inputs
    cnn_dim = n_filters[0] * ts_shape[0]/(2**len(n_filters))
    # ts_len = ts_shape[0]
    ts_dim = ts_shape[1]
    x = Dense(cnn_dim, activation='relu')(x)
    x = Reshape((-1, n_filters[0]))(x)
    for f in n_filters[1:]:
        x = Conv1DTranspose(f, k_size, strides=2, padding='same', activation='relu')(x)
    x = Conv1DTranspose(ts_dim, k_size, strides=2, padding='same', activation='relu')(x)
    x = Flatten()(x)
    x = Dense(ts_shape[0])(x)
    outputs = Reshape(ts_shape)(x)
    return Model(inputs, outputs, name='seq_decoder')

def ImageCNNEnc(input_shape, latent_dim, n_filters, k_size, dropout=0.0):
    inputs = Input(shape=input_shape)
    x = inputs
    for f in n_filters:
        x = Conv2D(f, k_size, padding='same', strides=1, activation='relu')(x)
        x = Dropout(dropout)(x)
        x = MaxPooling2D(2)(x)
    # x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    outputs = Dense(latent_dim)(x)
    return Model(inputs, outputs, name='img_encoder')

def ImageCNNEncMp(input_shape, latent_dim, n_filters, k_size, dropout=0.0):
    inputs = Input(shape=input_shape)
    x = inputs
    for f in n_filters:
        x = Conv2D(f, k_size, padding='same', strides=2, activation='relu')(x)
        x = Dropout(dropout)(x)
        # x = MaxPooling2D(2)(x)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    outputs = Dense(latent_dim)(x)
    return Model(inputs, outputs, name='img_encoder')

def ImageCNNEncNp(input_shape, latent_dim, n_filters, k_size, dropout=0.0):
    inputs = Input(shape=input_shape)
    x = inputs
    for f in n_filters:
        x = Conv2D(f, k_size, padding='same', strides=2, activation='relu')(x)
        x = Dropout(dropout)(x)
        # x = MaxPooling2D(2)(x)
    # x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    # x = Dense(128, activation='relu')(x)
    outputs = Dense(latent_dim)(x)
    return Model(inputs, outputs, name='img_encoder')

def ImageCNNDec(input_shape, img_shape, n_filters, k_size):
    inputs = Input(shape=input_shape)
    x = inputs
    cnn_in = int(img_shape[0] / (2**len(n_filters)))
    cnn_dim = int(n_filters[0] * cnn_in * cnn_in)
    img_dim = img_shape[-1]
    x = Dense(cnn_dim, activation='relu')(x)
    x = Reshape((cnn_in, cnn_in, n_filters[0]))(x)
    for f in n_filters[1:]:
        x = Conv2DTranspose(f, k_size, strides=2, padding='same', activation='relu')(x)
    outputs = Conv2DTranspose(img_dim, k_size, strides=2, padding='same', activation='tanh')(x)
    return Model(inputs, outputs, name='img_decoder')

def Discriminator(input_shape, hidden_unit, dropout=0.3):
    inputs = Input(input_shape)
    x = Dense(hidden_unit, activation='relu')(inputs)
    x = Dropout(dropout)(x)
    x = Dense(hidden_unit, activation='relu')(x)
    x = Dropout(dropout)(x)
    outputs = Dense(1)(x)
    return Model(inputs, outputs, name='discriminator')