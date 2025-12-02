# https://colab.research.google.com/drive/1QiT8V0vp6BQtbqjLn-oe58PA99MDDve1?usp=sharing

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from tensorflow.keras.callbacks import EarlyStopping

# Параметры
max_words = 10000
max_len = 200
batch_size = 64
epochs = 5

# 1. Загрузка данных
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# 2. Паддинг
x_train = sequence.pad_sequences(x_train, maxlen=max_len, padding='pre')
x_test = sequence.pad_sequences(x_test, maxlen=max_len, padding='pre')

# 3. Создание модели
def create_model(optimizer='adam', lr=0.001):

    model = Sequential()

    # Embedding
    model.add(Embedding(max_words, 128, input_length=max_len))

    # Стек рекуррентных слоев
    model.add(LSTM(32, return_sequences=True, dropout=0.2))
    model.add(Bidirectional(LSTM(32, dropout=0.2)))

    # Выходной слой
    model.add(Dense(1, activation='sigmoid'))

    # Оптимизатор
    if optimizer == 'adam':
        opt = Adam(lr)
    elif optimizer == 'rmsprop':
        opt = RMSprop(lr)
    else:
        opt = SGD(lr, momentum=0.9, nesterov=True)

    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

# Оптимизаторы и LR
optimizers = {
    'Adam': [0.01, 0.001, 0.0001],
    'RMSProp': [0.01, 0.001, 0.0001],
    'SGD_Nesterov': [0.01, 0.001,0.0001]
}

early_stopping = EarlyStopping(monitor='val_loss', patience=2, restore_best_weights=True)

results = {}

for opt_name, lr_list in optimizers.items():

    best_acc = 0
    best_lr = None

    print(f"\n=== {opt_name} ===")

    for lr in lr_list:
        print(f"\nLR = {lr}")

        model = create_model(
            optimizer=('sgd' if opt_name=='SGD_Nesterov' else opt_name.lower()),
            lr=lr
        )

        history = model.fit(
            x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=1
        )

        test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
        print(f"Точность на тесте: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            best_lr = lr

        results[(opt_name, lr)] = test_acc

    print(f"\nЛучший LR для {opt_name}: {best_lr}, точность: {best_acc*100:.2f}%")

print("\n=== ИТОГИ ===")
for (name, lr), acc in results.items():
    print(f"{name} lr={lr} → acc={acc*100:.2f}%")


Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb.npz
17464789/17464789 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step

=== Adam ===

LR = 0.01
Epoch 1/5
/usr/local/lib/python3.12/dist-packages/keras/src/layers/core/embedding.py:97: UserWarning: Argument `input_length` is deprecated. Just remove it.
  warnings.warn(
313/313 ━━━━━━━━━━━━━━━━━━━━ 107s 320ms/step - accuracy: 0.6996 - loss: 0.5574 - val_accuracy: 0.8394 - val_loss: 0.3753
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 95s 303ms/step - accuracy: 0.8740 - loss: 0.3142 - val_accuracy: 0.8654 - val_loss: 0.3332
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 141s 301ms/step - accuracy: 0.9310 - loss: 0.1811 - val_accuracy: 0.8674 - val_loss: 0.3358
Epoch 4/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 95s 304ms/step - accuracy: 0.9572 - loss: 0.1233 - val_accuracy: 0.8618 - val_loss: 0.3747
Точность на тесте: 85.88%

LR = 0.001
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 99s 298ms/step - accuracy: 0.7002 - loss: 0.5404 - val_accuracy: 0.8738 - val_loss: 0.3067
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 141s 296ms/step - accuracy: 0.9031 - loss: 0.2453 - val_accuracy: 0.8780 - val_loss: 0.3176
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 92s 296ms/step - accuracy: 0.9397 - loss: 0.1679 - val_accuracy: 0.8594 - val_loss: 0.3483
Точность на тесте: 86.76%

LR = 0.0001
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 101s 301ms/step - accuracy: 0.5412 - loss: 0.6857 - val_accuracy: 0.7944 - val_loss: 0.4995
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 93s 298ms/step - accuracy: 0.8155 - loss: 0.4433 - val_accuracy: 0.8500 - val_loss: 0.3699
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 99s 317ms/step - accuracy: 0.8827 - loss: 0.3053 - val_accuracy: 0.8636 - val_loss: 0.3259
Epoch 4/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 95s 303ms/step - accuracy: 0.9127 - loss: 0.2425 - val_accuracy: 0.8714 - val_loss: 0.3142
Epoch 5/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 140s 297ms/step - accuracy: 0.9311 - loss: 0.1965 - val_accuracy: 0.8710 - val_loss: 0.3304
Точность на тесте: 86.47%

Лучший LR для Adam: 0.001, точность: 86.76%

=== RMSProp ===

LR = 0.01
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 101s 304ms/step - accuracy: 0.6244 - loss: 0.6301 - val_accuracy: 0.7808 - val_loss: 0.4695
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 141s 302ms/step - accuracy: 0.8479 - loss: 0.3618 - val_accuracy: 0.8274 - val_loss: 0.4454
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 93s 298ms/step - accuracy: 0.9095 - loss: 0.2312 - val_accuracy: 0.8762 - val_loss: 0.3003
Epoch 4/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 144s 304ms/step - accuracy: 0.9436 - loss: 0.1570 - val_accuracy: 0.8766 - val_loss: 0.3926
Epoch 5/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 94s 300ms/step - accuracy: 0.9645 - loss: 0.1044 - val_accuracy: 0.8794 - val_loss: 0.3570
Точность на тесте: 87.82%

LR = 0.001
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 102s 307ms/step - accuracy: 0.5985 - loss: 0.6365 - val_accuracy: 0.8182 - val_loss: 0.4149
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 146s 322ms/step - accuracy: 0.8400 - loss: 0.3855 - val_accuracy: 0.8736 - val_loss: 0.3093
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 94s 300ms/step - accuracy: 0.8849 - loss: 0.2913 - val_accuracy: 0.8752 - val_loss: 0.3123
Epoch 4/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 99s 315ms/step - accuracy: 0.9064 - loss: 0.2422 - val_accuracy: 0.8694 - val_loss: 0.3517
Точность на тесте: 87.20%

LR = 0.0001
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 101s 303ms/step - accuracy: 0.5163 - loss: 0.6928 - val_accuracy: 0.5644 - val_loss: 0.6916
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 143s 304ms/step - accuracy: 0.5997 - loss: 0.6882 - val_accuracy: 0.6986 - val_loss: 0.5982
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 99s 316ms/step - accuracy: 0.7299 - loss: 0.5464 - val_accuracy: 0.8266 - val_loss: 0.4141
Epoch 4/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 137s 300ms/step - accuracy: 0.8313 - loss: 0.3998 - val_accuracy: 0.8582 - val_loss: 0.3561
Epoch 5/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 97s 309ms/step - accuracy: 0.8663 - loss: 0.3287 - val_accuracy: 0.8616 - val_loss: 0.3361
Точность на тесте: 85.90%

Лучший LR для RMSProp: 0.01, точность: 87.82%

=== SGD_Nesterov ===

LR = 0.01
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 98s 297ms/step - accuracy: 0.4942 - loss: 0.6936 - val_accuracy: 0.4938 - val_loss: 0.6935
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 91s 289ms/step - accuracy: 0.5078 - loss: 0.6932 - val_accuracy: 0.5062 - val_loss: 0.6927
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 95s 305ms/step - accuracy: 0.5109 - loss: 0.6925 - val_accuracy: 0.5426 - val_loss: 0.6915
Epoch 4/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 92s 293ms/step - accuracy: 0.5271 - loss: 0.6915 - val_accuracy: 0.5194 - val_loss: 0.6899
Epoch 5/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 92s 293ms/step - accuracy: 0.5547 - loss: 0.6865 - val_accuracy: 0.5918 - val_loss: 0.6629
Точность на тесте: 58.50%

LR = 0.001
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 105s 314ms/step - accuracy: 0.5015 - loss: 0.6932 - val_accuracy: 0.4926 - val_loss: 0.6933
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 135s 293ms/step - accuracy: 0.5058 - loss: 0.6932 - val_accuracy: 0.4842 - val_loss: 0.6932
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 91s 289ms/step - accuracy: 0.5041 - loss: 0.6931 - val_accuracy: 0.4932 - val_loss: 0.6933
Epoch 4/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 91s 292ms/step - accuracy: 0.5039 - loss: 0.6932 - val_accuracy: 0.4936 - val_loss: 0.6933
Точность на тесте: 50.45%

LR = 0.0001
Epoch 1/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 97s 293ms/step - accuracy: 0.5033 - loss: 0.6931 - val_accuracy: 0.5028 - val_loss: 0.6931
Epoch 2/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 91s 289ms/step - accuracy: 0.5074 - loss: 0.6930 - val_accuracy: 0.5020 - val_loss: 0.6931
Epoch 3/5
313/313 ━━━━━━━━━━━━━━━━━━━━ 143s 292ms/step - accuracy: 0.5066 - loss: 0.6931 - val_accuracy: 0.5014 - val_loss: 0.6931
Точность на тесте: 50.10%

Лучший LR для SGD_Nesterov: 0.01, точность: 58.50%

=== ИТОГИ ===
Adam lr=0.01 → acc=85.88%
Adam lr=0.001 → acc=86.76%
Adam lr=0.0001 → acc=86.47%
RMSProp lr=0.01 → acc=87.82%
RMSProp lr=0.001 → acc=87.20%
RMSProp lr=0.0001 → acc=85.90%
SGD_Nesterov lr=0.01 → acc=58.50%
SGD_Nesterov lr=0.001 → acc=50.45%
SGD_Nesterov lr=0.0001 → acc=50.10%

