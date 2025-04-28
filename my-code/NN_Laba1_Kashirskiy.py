import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

num_samples = 1000
x1 = np.random.rand(num_samples) * 10
x2 = np.random.rand(num_samples) * 10
x3 = np.random.rand(num_samples) * 10

y = x1 + 2*x2 + 3*x3

X = np.column_stack((x1, x2, x3))

model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(units=1, input_shape=[3])
])

model.compile(optimizer='sgd', loss='mean_squared_error')

history = model.fit(X, y, epochs=400)

plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()

test_input = np.array([[1, 2, 3]])
prediction = model.predict(test_input)
print(f"Предсказание для x1=1, x2=2, x3=3: {prediction}")

print(model.get_weights())

#https://colab.research.google.com/drive/1n4etfEf6ZAdyUsE23--m3HWCs15LLXJx?usp=drive_link