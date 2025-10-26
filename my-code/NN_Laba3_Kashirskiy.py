# Практическая работа №3
# Исследование сверточной нейронной сети для классификации полноцветных изображений из БД CIFAR-10
# Студент: Каширский Н.Е. Ум-242
# Ссылка на репозиторий - https://github.com/kolyan1922/Tasks2/blob/main/my-code/NN_Laba3_Kashirskiy.py
# Ссылка на google colab - https://colab.research.google.com/drive/1O8SBm4kEl4rmKbY_TZrtnVy9IXLfcjFZ?usp=sharing
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import time

# Парамеры
EPOCHS = 50
BATCH_SIZE = 128
VAL_RATIO = 0.1
PATIENCE = 7
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# Проверка GPU
print("GPU available:", tf.config.list_physical_devices('GPU'))

# Загрузка и нормализация CIFAR-10
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
y_train, y_test = y_train.flatten(), y_test.flatten()

x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Разделение на тренировку и валидацию
num_val = int(len(x_train) * VAL_RATIO)
x_val, y_val = x_train[:num_val], y_train[:num_val]
x_train, y_train = x_train[num_val:], y_train[num_val:]

# Data Augmentation — аугментация изображений
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    zoom_range=0.1
)
datagen.fit(x_train)


# Функция для построения свёрточной нейронной сети на Keras
def build_customnet(input_shape=(32, 32, 3), num_classes=10):
    model = keras.Sequential([
        layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),

        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),

        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

# Функция обучения и оценки
def train_and_evaluate(optimizer, lr, name):
    print(f"\n Обучение с {name} (lr={lr})")
    model = build_customnet()
    model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    callbacks = [
        keras.callbacks.EarlyStopping(patience=PATIENCE, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=3)
    ]

    start = time.time()
    history = model.fit(
        datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
        validation_data=(x_val, y_val),
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=2
    )
    duration = time.time() - start

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"* Тестовая точность ({name}, lr={lr}): {test_acc:.4f}")
    return history, test_acc, duration

# Подбор оптимизаторов и параметров
optimizers_to_test = {
    "Adam": [1e-3, 5e-4],
    "RMSprop": [1e-3, 5e-4],
    "SGD+Nesterov": [0.01, 0.005]
}

results = {}
histories = {}

for name, lrs in optimizers_to_test.items():
    best_acc = 0
    best_lr = None
    for lr in lrs:
        if name == "Adam":
            opt = keras.optimizers.Adam(learning_rate=lr)
        elif name == "RMSprop":
            opt = keras.optimizers.RMSprop(learning_rate=lr)
        else:
            opt = keras.optimizers.SGD(learning_rate=lr, momentum=0.9, nesterov=True)

        hist, acc, dur = train_and_evaluate(opt, lr, name)
        key = f"{name} (lr={lr})"
        histories[key] = hist
        results[key] = (acc, dur)
        if acc > best_acc:
            best_acc = acc
            best_lr = lr
    print(f"** Лучший lr для {name}: {best_lr}, точность = {best_acc:.4f}")

# Визуализация графиков
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
for name, hist in histories.items():
    plt.plot(hist.history['val_accuracy'], label=name)
plt.title('Validation Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
for name, hist in histories.items():
    plt.plot(hist.history['val_loss'], label=name)
plt.title('Validation Loss per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

# Итоговая таблица
print("\n Итоговые результаты:")
print(f"{'Оптимизатор':20s} | {'Точность':10s} | {'Время (сек)':10s}")
print("-"*50)
for name, (acc, dur) in results.items():
    print(f"{name:20s} | {acc:.4f}     | {dur:8.1f}")

