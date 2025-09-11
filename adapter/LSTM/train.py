import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Concatenate
from tensorflow.keras.metrics import Recall
from tensorflow.keras import regularizers, saving, callbacks


history_seconds = 60
next_prediction_period = 10
input_shape = history_seconds // next_prediction_period

BINARY_THRESHOLD = 0.4

dir = os.path.dirname(__file__)


def get_x_y(data):
    seq = []
    target = []
    y = []
    for i in range(0, len(data) - history_seconds, history_seconds // 2):
        t = data[i:i + history_seconds]
        temp_x = []
        for j in range(0, len(t), next_prediction_period):
            temp_x.append(max(t[j:j + next_prediction_period]))
        
        next_prediction_value = max(data[i+history_seconds:i+history_seconds+next_prediction_period])
        for sample_target in [min(temp_x), (min(temp_x) + max(temp_x)) // 2, max(temp_x), max(temp_x) + (max(temp_x) - min(temp_x))]:
            seq.append(temp_x)
            target.append(sample_target)
            if next_prediction_value > sample_target:
                y.append(1)
            else:
                y.append(0)
    return [tf.convert_to_tensor(seq, dtype=tf.int32), tf.convert_to_tensor(target, dtype=tf.int32)], tf.convert_to_tensor(y)


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
        train_x,
        tf.convert_to_tensor(np.array(train_y), dtype=tf.int32),
        test_x,
        tf.convert_to_tensor(np.array(test_y), dtype=tf.int32)
    )


def create_model():
    sequence_input = Input(shape=(input_shape, 1), name="sequence_input")
    target_input = Input(shape=(1,), name="target_input")
    x = LSTM(25, activation="relu", kernel_regularizer=regularizers.L1(0.00001))(sequence_input)
    merged = Concatenate()([x, target_input])
    output = Dense(1, activation="sigmoid")(merged)
    model = Model(inputs=[sequence_input, target_input], outputs=output)
    return model


def my_train():
    tf.random.set_seed(7)
    train_x, train_y, test_x, test_y = get_data()
    print("shapes")
    print(len(train_x),)
    print(train_y.shape)
    print(len(test_x),)
    print(test_y.shape)
    model = create_model()
    print("model summary")
    print(model.summary())
    print("now compiling")
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", Recall()])
    print("callback definition")
    callback = callbacks.EarlyStopping(monitor="val_recall", mode="max", patience=5, restore_best_weights=True)
    print("now fitting")
    model.fit(train_x, train_y, epochs=50, batch_size=16, validation_data=(test_x, test_y), callbacks=[callback])
    print("now predicting")
    predictions = model.predict(test_x)
    
    # Lower the threshold to have a better recall (worse precision!)
    
    y_pred_classes = (predictions >= BINARY_THRESHOLD).astype(int)
    tp = 0
    fn = 0
    for i in range(len(test_y)):
        y = test_y[i]
        pred = y_pred_classes[i]
        if y == pred == 1:
            tp += 1
        if y == 1 and pred == 0:
            fn += 1
        
    print("recall", (tp / (tp + fn)))

    model.save(f'{dir}/lstm_binary.keras')


def my_test():
    model = saving.load_model(f'{dir}/lstm_binary.keras')
    with open("workload2.txt", "r") as f:
        workload = f.readlines()
    workload = workload[0].split()
    workload = list(map(lambda x: int(x) // 8, workload))
    workload = list(filter(lambda x: x != 0, workload))

    minute = 60
    hour = 60 * 60
    day = hour * 24
    test_idx = 15 * day

    # test_data = workload[test_idx:test_idx + 20 * minute]
    # test_data = workload[test_idx:]
    test_data2 = workload[80:20*minute+80]
    test_data = []
    for i in range(0, len(test_data2) -2, 2):
        test_data.append(int((test_data2[i] + test_data2[i+1]) / 2))

    test_x, test_y = get_x_y(test_data)

    predictions = model.predict(test_x)
    y_pred_classes = (predictions >= BINARY_THRESHOLD).astype(int)
    print(max(test_data[:120]))
    # plt.plot(test_data)
    # plt.show()
    # plt.scatter(list(range(len(test_y))), list(test_y), label="real values")
    # plt.scatter(list(range(len(test_y))), list(y_pred_classes), label="predictions")
    # plt.xlabel("time (minute)")
    # plt.ylabel("load (RPS)")
    # plt.legend()
    # plt.show()
    tp = 0
    fn = 0
    fp = 0
    for i in range(len(test_y)):
        y = test_y[i]
        pred = y_pred_classes[i]
        if y == pred == 1:
            tp += 1
        if y == 0 and pred == 1:
            fp += 1
        if y == 1 and pred == 0:
            fn += 1
        
    print("precision", (tp / (tp + fp)))
    print("recall", (tp / (tp + fn)))


if __name__ == "__main__":
    # my_train()
    my_test()
    # model = saving.load_model(f'{dir}/lstm_binary.keras')
    # h = tf.convert_to_tensor(np.array([34, 31, 22, 29, 32, 31]).reshape((-1, 6, 1)), dtype=tf.float32)
    # preds = model.predict([h, tf.convert_to_tensor([50])])
    # y_pred_classes = (preds >= BINARY_THRESHOLD).astype(int)
    # print(bool(y_pred_classes))
