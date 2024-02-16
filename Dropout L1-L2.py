import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop, Adam
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt

learning_rate = 0.001
epochs = 200
batch_size = 120

import mlflow
mlflow.tensorflow.autolog()
dataset=mnist.load_data()

(x_train, y_train), (x_test, y_test) = dataset

x_trainv = x_train.reshape(60000, 784)
x_testv = x_test.reshape(10000, 784)

x_trainv = x_trainv.astype('float32')
x_testv = x_testv.astype('float32')
x_trainv /= 1700
x_testv /= 1700

num_classes=10
y_trainc = keras.utils.to_categorical(y_train, num_classes)
y_testc = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Dense(500, activation='sigmoid', input_shape=(784,), kernel_regularizer=regularizers.l1_l2(l1=0.0000001, l2=0.0000001)))
model.add(Dropout(0.05))
model.add(Dense(250, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.0000001, l2=0.0000001)))
model.add(Dropout(0.05))
model.add(Dense(100, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.0000001, l2=0.0000001)))
model.add(Dropout(0.05))
model.add(Dense(75, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.0000001, l2=0.0000001)))
model.add(Dropout(0.05))
model.add(Dense(30, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.0000001, l2=0.0000001)))
model.add(Dropout(0.05))
model.add(Dense(num_classes, activation='sigmoid', kernel_regularizer=regularizers.l1_l2(l1=0.0000001, l2=0.0000001)))
model.summary()
model.compile(loss='categorical_crossentropy',optimizer=Adam(learning_rate=learning_rate),metrics=['accuracy'])
history = model.fit(x_trainv, y_trainc,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_testv, y_testc)
                    )

score = model.evaluate(x_testv, y_testc, verbose=1)
print(score)
a=model.predict(x_testv)

print(a.shape)
print(a[1])
print("resultado correcto:")
print(y_testc[1])

# Graficar el sobreajuste
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

model.save("red.h5")
exit()
modelo_cargado = tf.keras.models.load_model('red.h5')
