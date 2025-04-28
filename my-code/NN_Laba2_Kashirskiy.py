import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

num_pixels = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape((x_train.shape[0], num_pixels))
x_test = x_test.reshape((x_test.shape[0], num_pixels))

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

num_classes = y_test.shape[1]

def create_and_train_model(activation_function='relu', optimizer='adam'):
    model = keras.Sequential([
        keras.layers.Dense(128, activation=activation_function, input_shape=(num_pixels,)),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer=optimizer,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    history = model.fit(x_train, y_train,
                        epochs=10,
                        batch_size=32,
                        validation_split=0.2,
                        verbose=0)
    return history, model

activation_functions = ['linear', 'sigmoid', 'tanh', 'relu']

histories = {}
models = {}
test_accuracies = {}

for activation in activation_functions:
    print(f"Обучение: {activation}")
    history, model = create_and_train_model(activation_function=activation)
    histories[activation] = history
    models[activation] = model
    _, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
    test_accuracies[activation] = test_accuracy
    print(f"Точность: {test_accuracy:.4f}")
    
    plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
for activation, history in histories.items():
    plt.plot(history.history['accuracy'], label=f'{activation} Train')
    plt.plot(history.history['val_accuracy'], label=f'{activation} Val')
plt.title('Accuracy vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
for activation, history in histories.items():
    plt.plot(history.history['loss'], label=f'{activation} Train')
    plt.plot(history.history['val_loss'], label=f'{activation} Val')
plt.title('Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

print("\n Точность на тесте:")
for activation, accuracy in test_accuracies.items():
    print(f"{activation}: {accuracy:.4f}")

best_activation = max(test_accuracies, key=test_accuracies.get)
best_model = models[best_activation]
print(f"\n Лучшая функция активации: {best_activation} (точность: {test_accuracies[best_activation]:.4f})")

best_model.summary()

# https://colab.research.google.com/drive/1xstBhNGKGF5a4xIu3wlumcXAfSzraYHz?usp=drive_link