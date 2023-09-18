from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.utils import to_categorical
import tensorflow as tf
import numpy as np
import pickle

from Check_ropa import Check

def guardar_modelo(history, model):
    model.save_weights(CHECKPOINT_MODEL_PATH)

    with open(CHECKPOINT_HISTORY_PATH, 'wb') as file_pi:
        pickle.dump(history.history, file_pi)

def cargar_modelo(model):
    model.load_weights(CHECKPOINT_MODEL_PATH)

    with open(CHECKPOINT_HISTORY_PATH, "rb") as file_pi:
        history = pickle.load(file_pi)

    return model, history

def generar_datos(ck):
    #CREAR DATASET
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    #ck.check_dataset(x_train, y_train)

    x_train, y_train, x_test, y_test = formatear_datos(x_train, y_train, x_test, y_test)
    #ck.check_formato_datos(train_features, train_y)

    return x_train, y_train, x_test, y_test

def formatear_x(x):
    x = np.array(x)
    x = x.reshape(x.shape[0], IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_CHANNELS)
    x = np.float32(x)
    x /= 255

    return x

def formatear_y(y):
    y = np.array(y)
    y = np.float32(y)
    y = to_categorical(y)

    return y

def formatear_datos(x_train, y_train, x_test, y_test):
    #train
    x_train = formatear_x(x_train)
    x_test = formatear_x(x_test)

    y_train = formatear_y(y_train)
    y_test = formatear_y(y_test)

    return x_train, y_train, x_test, y_test

def predict(model, test_x):
    y_pred = []
    y_pred_aux = model.predict(test_x)
    for i in y_pred_aux:
        y_pred.append(np.argmax(i))
    y_pred = formatear_y(y_pred)

    return y_pred

def evaluar(model, x_test, y_test):
    test_eval = model.evaluate(x_test, y_test, verbose=1)
    print('Test loss: ', test_eval[0])
    print('Test accuracy: ', test_eval[1])

def test(model, ck):
    _, _, x_test, y_test = generar_datos(ck)

    #ck.check_plot_labels(y_test)
    
    evaluar(model, x_test, y_test)

    #Prediccion
    y_pred = predict(model, x_test)
    #ck.check_prediction(y_pred)
    ck.check_matriz_confusion(y_test, y_pred)
    ck.check_metricas(y_test, y_pred)

def modelo():
    #DEFINIR MODELO
    model = Sequential()
    model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=(IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS), padding='same'))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    model.add(Dropout(0.30))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.50))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(2,2), padding='same'))
    model.add(Dropout(0.50))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(NUM_CLASES, activation='sigmoid'))

    model.compile(loss='categorical_crossentropy', optimizer= tf.keras.optimizers.Adam(), metrics=['accuracy'])

    return model

def callbacks():
    #AÑADIR PARA OPTIMIZACION (EVITAR VUELTAS DE MÁS)
    from keras.callbacks import EarlyStopping, ReduceLROnPlateau
    earlystop = EarlyStopping(patience=PATIENCE)
    learning_rate_reduction = ReduceLROnPlateau(monitor='val_accuracy',
                                                patience=2,
                                                verbose=1,
                                                factor=0.5,
                                                min_lr=0.00001)
    callbacks = [earlystop, learning_rate_reduction]

    return callbacks

def entrenamiento(x_train, x_valid, y_train, y_valid, model, callbacks):
    #ENTRENAMIENTO
    history = model.fit(
        x_train,
        y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        verbose=2,
        validation_data=(x_valid, y_valid),
        callbacks=callbacks
    )

    return history, model

def generar_modelo(ck):
    #Se descargan los datos y de formatean
    x_train, y_train, _, _ = generar_datos(ck)
    #ck.check_plot_labels(y_train)

    #Preparacion para modelo
    x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.20, random_state=42)
    #ck.check_prep_modelo(x_train, x_valid, y_train, y_valid)

    #Definir modelo
    model = modelo()
    ck.check_model(model)

    #Optimizacion
    cb = callbacks()
    #cb = None

    #Entrenamiento
    history, model = entrenamiento(x_train, x_valid, y_train, y_valid, model, cb)

    guardar_modelo(history, model)

    return history.history, model

def recuperar_modelo(ck):
    #Definir modelo
    model = modelo()

    model, history = cargar_modelo(model)
    ck.check_model(model)

    return model, history

def main():
    ck = Check(EPOCHS)

    if GENERAR_MODELO:
        #Genera el modelo de 0
        history, model = generar_modelo(ck)
    else:
        #Carga un modelo previamente generado y guardado
        model, history = recuperar_modelo(ck)

    #ck.check_perdida(history)
    #ck.check_metricas_plot(history)

    test(model, ck)

    print("fin")
    

IMAGE_WIDTH = 28 
IMAGE_HEIGHT = 28 
IMAGE_SIZE = (IMAGE_WIDTH, IMAGE_HEIGHT)
IMAGE_CHANNELS = 1

BATCH_SIZE = 30
NUM_CLASES = 10
EPOCHS = 20
PATIENCE = 10

GENERAR_MODELO = False
#MODEL_NUM = 5
#SAVE_COUNT = 4

INI_FILES = "Unidad 2. Deep Learning/Practica ropa/dogs-vs-cats"
TRAIN_DIR_UNZIP = "Unidad 2. Deep Learning/Practica ropa/dogs-vs-cats/train.zip"
TEST_DIR_UNZIP = "Unidad 2. Deep Learning/Practica ropa/dogs-vs-cats/test1.zip"

TRAIN_DIR = "Unidad 2. Deep Learning/Practica ropa/dogs-vs-cats/train"
TEST_DIR = "Unidad 2. Deep Learning/Practica ropa/dogs-vs-cats/test"
TEST_DIR_PARCIAL = "Unidad 2. Deep Learning/Practica ropa/dogs-vs-cats/test_parcial"

CHECKPOINT_MODEL_PATH = "Unidad 2. Deep Learning/Practica ropa/checkpoint/model.h5" #"_" + str(MODEL_NUM) + "_" + str(SAVE_COUNT) + ".h5"
CHECKPOINT_HISTORY_PATH = "Unidad 2. Deep Learning/Practica ropa/checkpoint/history.npy" #"_" + str(MODEL_NUM) + "_" + str(SAVE_COUNT) + ".npy"

FEATURES_PATH = "Unidad 2. Deep Learning/Practica ropa/checkpoint/features.npy"
LABELS_PATH = "Unidad 2. Deep Learning/Practica ropa/checkpoint/labels.npy"
FEATURES_TEST_PATH = "Unidad 2. Deep Learning/Practica ropa/checkpoint/features_test_828.npy"
LABELS_TEST_PATH = "Unidad 2. Deep Learning/Practica ropa/checkpoint/labels_test_828.npy"


main()