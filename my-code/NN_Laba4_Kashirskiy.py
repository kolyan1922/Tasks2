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
