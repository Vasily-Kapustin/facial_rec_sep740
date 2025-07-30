import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, add, Activation, MaxPooling2D, Conv2D, Flatten, Dropout
from keras.layers import LeakyReLU
from sklearn.model_selection import train_test_split
import tensorflow.keras.backend as K
import cv2
import visualkeras

from Generators import *
from PlotMetrics import *
from DataPipeline import get_data, FaceImageAugmentor


def plot_gallery(images, titles, h, w, n_row=3, n_col=3):
    """Helper function to plot a gallery of portraits"""
    plt.figure(figsize=(1.8 * n_col, 2.4 * n_row))
    plt.subplots_adjust(bottom=0, left=.01, right=.99, top=.90, hspace=.35)
    i=0
    k=0
    t=[]
    while(k<6):
        plt.subplot(n_row, n_col, k + 1)
        if(titles[i] in t):
          i=i+1
          continue
        else:
          plt.imshow(images[i].reshape((h, w)), cmap=plt.cm.gray)
          plt.title(titles[i], size=12)
          plt.xticks(())
          plt.yticks(())
          t.append(titles[i])
          k=k+1
        i=i+1


def title(y_pred, y_test, target_names, i):
    pred_name = target_names[y_pred[i]].rsplit(' ', 1)[-1]
    true_name = target_names[y_test[i]].rsplit(' ', 1)[-1]
    return 'predicted: %s\ntrue:      %s' % (pred_name, true_name)


def main():
    X, y, target_names = get_data(min_pics=50)
    num_classes = len(target_names)
    y = tf.keras.utils.to_categorical(y, num_classes=num_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='linear', padding='valid', input_shape=X.shape[1:]))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D((2, 2), padding='valid'))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='linear', padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, (3, 3), activation='linear', padding='valid'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(MaxPooling2D(pool_size=(2, 2), padding='valid'))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense(64, activation='linear'))
    model.add(LeakyReLU(alpha=0.1))
    model.add(Dropout(0.3))
    model.add(Dense((num_classes), activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

    visualkeras.layered_view(model, to_file='output.png', legend=True)
    callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=10)

    H1 = model.fit(X_train, y_train, epochs=150, batch_size=100, verbose=1, validation_split=0.1, callbacks=[callback])
    plot_training_history(H1,"Classifier")
    y_pred = model.predict(X_test)
    y_test = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)
    prediction_titles = [title(y_pred, y_test, target_names, i)
                         for i in range(y_pred.shape[0])]

    plot_gallery(X_test, prediction_titles, 64, 64)

    plt.show()

if __name__ == '__main__':
    main()