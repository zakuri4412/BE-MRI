import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model

IMG_SIZE = 128
INPUT_SHAPE = (IMG_SIZE, IMG_SIZE, 1)

def attention_gate(x, g, inter_channels):
    theta_x = Conv2D(inter_channels, 1, padding='same')(x)
    phi_g   = Conv2D(inter_channels, 1, padding='same')(g)

    add_xg = Add()([theta_x, phi_g])
    add_xg = Activation('relu')(add_xg)

    psi = Conv2D(1, 1, padding='same')(add_xg)
    psi = Activation('sigmoid')(psi)

    return Multiply()([x, psi])


def build_attention_unet_binary(input_shape):
    inputs = Input(input_shape)

    c1 = Conv2D(32, 3, activation='relu', padding='same')(inputs)
    c1 = Conv2D(32, 3, activation='relu', padding='same')(c1)
    p1 = MaxPooling2D()(c1)

    c2 = Conv2D(64, 3, activation='relu', padding='same')(p1)
    c2 = Conv2D(64, 3, activation='relu', padding='same')(c2)
    p2 = MaxPooling2D()(c2)

    c3 = Conv2D(128, 3, activation='relu', padding='same')(p2)
    c3 = Conv2D(128, 3, activation='relu', padding='same')(c3)
    p3 = MaxPooling2D()(c3)

    c4 = Conv2D(256, 3, activation='relu', padding='same')(p3)
    c4 = Conv2D(256, 3, activation='relu', padding='same')(c4)
    p4 = MaxPooling2D()(c4)

    bn = Conv2D(512, 3, activation='relu', padding='same')(p4)
    bn = Conv2D(512, 3, activation='relu', padding='same')(bn)

    u4 = UpSampling2D()(bn)
    att4 = attention_gate(c4, u4, 256)
    m4 = Concatenate()([u4, att4])
    c5 = Conv2D(256, 3, activation='relu', padding='same')(m4)
    c5 = Conv2D(256, 3, activation='relu', padding='same')(c5)

    u3 = UpSampling2D()(c5)
    att3 = attention_gate(c3, u3, 128)
    m3 = Concatenate()([u3, att3])
    c6 = Conv2D(128, 3, activation='relu', padding='same')(m3)
    c6 = Conv2D(128, 3, activation='relu', padding='same')(c6)

    u2 = UpSampling2D()(c6)
    att2 = attention_gate(c2, u2, 64)
    m2 = Concatenate()([u2, att2])
    c7 = Conv2D(64, 3, activation='relu', padding='same')(m2)
    c7 = Conv2D(64, 3, activation='relu', padding='same')(c7)

    u1 = UpSampling2D()(c7)
    att1 = attention_gate(c1, u1, 32)
    m1 = Concatenate()([u1, att1])
    c8 = Conv2D(32, 3, activation='relu', padding='same')(m1)
    c8 = Conv2D(32, 3, activation='relu', padding='same')(c8)

    outputs = Conv2D(1, 1, activation='sigmoid')(c8)

    return Model(inputs, outputs)


model = build_attention_unet_binary((128,128,1))
model.load_weights("weights/model_ATTUNET.weights.h5")

