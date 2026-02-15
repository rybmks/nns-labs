import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
Y = np.array([0, 1, 1, 0])

def main():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(3, input_dim=2, activation="tanh"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
        loss="binary_crossentropy",
        metrics=['accuracy']
    )

    history = model.fit(X, Y, epochs=100, verbose=0)

    loss, accuracy = model.evaluate(X, Y, verbose=0)
    print("loss:", loss)
    print("accuracy:", accuracy)

    prediction = model.predict(X)
    for input, pred in zip(X, prediction):
        print(input, "Out rounded:", round(pred[0]), "Out:", pred[0])

    plt.figure()
    plt.plot(history.history['loss'])
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
