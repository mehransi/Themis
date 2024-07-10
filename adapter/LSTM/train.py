import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense
from tensorflow.keras import regularizers
from tensorflow.keras import saving

history_seconds = 60
next_prediction_period = 10
input_shape = history_seconds // next_prediction_period


def get_x_y(data):
    x = []
    y = []
    for i in range(0, len(data) - history_seconds, next_prediction_period):
        t = data[i:i + history_seconds]
        for j in range(0, len(t), next_prediction_period):
            x.append(max(t[j:j + next_prediction_period]))
        y.append(max(data[i+history_seconds:i+history_seconds+next_prediction_period]))
    return x, y


def get_data():
    with open('workload.txt', "r") as f:
        workload = f.readlines()
    workload = workload[0].split()
    workload = list(map(int, workload))

    workload = list(filter(lambda x: x != 0, workload))
    train_to_idx = 14 * 24 * 60 * 60
    workload_train = workload[:train_to_idx]
    workload_test = workload[train_to_idx:]

    train_x, train_y = get_x_y(workload_train)
    test_x, test_y = get_x_y(workload_test)
    return (
        tf.convert_to_tensor(np.array(train_x).reshape((-1, input_shape, 1)), dtype=tf.int32),
        tf.convert_to_tensor(np.array(train_y), dtype=tf.int32),
        tf.convert_to_tensor(np.array(test_x).reshape((-1, input_shape, 1)), dtype=tf.int32),
        tf.convert_to_tensor(np.array(test_y), dtype=tf.int32)
    )


def create_model():
    model = Sequential()
    model.add(Input(shape=(input_shape, 1)))
    model.add(LSTM(25, activation="relu", kernel_regularizer=regularizers.L1(0.00001)))
    model.add(Dense(1))
    return model


def my_train():
    tf.random.set_seed(7)
    train_x, train_y, test_x, test_y = get_data()
    print(train_x.shape)
    print(train_y.shape)
    print(test_x.shape)
    print(test_y.shape)
    model = create_model()
    print(model.summary())
    model.compile(optimizer="adam", loss="mse")
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode="min", min_delta=3, restore_best_weights=True)
    model.fit(train_x, train_y, epochs=50, batch_size=16, validation_data=(test_x, test_y), callbacks=[callback])
    predictions = model.predict(test_x)

    plt.plot(list(range(len(test_y))), list(test_y), label="real values")
    plt.plot(list(range(len(test_y))), list(predictions), label="predictions")
    plt.legend()
    plt.show()
    model.save('lstm60.keras')


def my_test():
    model = saving.load_model('lstm60.keras')
    with open("workload.txt", "r") as f:
        workload = f.readlines()
    workload = workload[0].split()
    workload = list(map(int, workload))
    workload = list(filter(lambda x: x != 0, workload))

    minute = 60
    hour = 60 * 60
    day = hour * 24
    test_idx = 15 * day

    test_data = workload[test_idx:test_idx + 20 * minute]

    test_x, test_y = get_x_y(test_data)

    test_x = tf.convert_to_tensor(np.array(test_x).reshape((-1, input_shape, 1)), dtype=tf.float32)

    prediction = model.predict(test_x)
    plt.plot(list(range(len(test_y))), list(test_y), label="real values")
    plt.plot(list(range(len(test_y))), list(prediction), label="predictions")
    plt.xlabel("time (minute)")
    plt.ylabel("load (RPS)")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # my_train()
    my_test()