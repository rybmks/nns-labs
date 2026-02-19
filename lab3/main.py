from pathlib import Path
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from PIL import Image

def download_dataset():
    working_dir = Path(__file__).parent / "dataset"
    working_dir.mkdir(parents=True, exist_ok=True)
    dataset_file = keras.utils.get_file(
        fname="mnist.npz",
        origin="https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz",
        cache_dir=str(working_dir.parent),
        cache_subdir=working_dir.name
    )
    return dataset_file

def predict_custom_image(model, image_path):
    img = Image.open(image_path).convert('L')  
    img = img.resize((28, 28))                 
    img_array = np.array(img)
    
    img_array = 255 - img_array 
    
    img_array = img_array.astype("float32") / 255
    img_array = img_array.reshape(1, 784)
    
    prediction = model.predict(img_array)
    result = np.argmax(prediction)
    
    plt.imshow(img, cmap='gray')
    plt.title(f"Розпізнана цифра: {result}")
    plt.axis('off')
    plt.show()
    return result

def main():
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data(path=str(download_dataset()))
    
    x_train = x_train.astype("float32") / 255 
    x_test = x_test.astype("float32") / 255 
    x_train = x_train.reshape(-1, 784) 
    x_test = x_test.reshape(-1, 784) 

    model = keras.Sequential([ 
        keras.layers.Input(shape=(784,)), 
        keras.layers.Dense(128, activation="relu"), 
        keras.layers.Dense(64, activation="relu"), 
        keras.layers.Dense(10, activation="softmax") 
    ])
    
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]) 

    model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test)) 

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    print("\nЗвіт про класифікацію (Precision, Recall, F-Score):")
    print(classification_report(y_test, y_pred_classes))

    cm = confusion_matrix(y_test, y_pred_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=np.arange(10))
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Матриця помилок")
    plt.show()

    custom_img_path = Path(__file__).parent / "digit5.jpg"
    if custom_img_path.exists():
        predict_custom_image(model, custom_img_path)
    else:
        print(f"\nФайл {custom_img_path.name} не знайдено")

if __name__ == "__main__":
    main()