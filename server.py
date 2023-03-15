from flask import Flask,jsonify,request
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score,precision_score,recall_score,accuracy_score
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical

app = Flask(__name__)

#Loading the iris dataset and training the model
Iris = load_iris()
X, y = Iris.data, Iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=3)
randomForest = RandomForestClassifier(n_estimators=10, criterion='entropy')
randomForest.fit(X_train,y_train)

#Defining and training the model
(X_train_CNN, y_train_CNN), (X_test_CNN, y_test_CNN) = mnist.load_data()
X_train_CNN = X_train_CNN.reshape((X_train_CNN.shape[0], 28, 28, 1))
X_test_CNN = X_test_CNN.reshape((X_test_CNN.shape[0], 28, 28, 1))
X_train_CNN = X_train_CNN.astype('float32') / 255.0
X_test_CNN = X_test_CNN.astype('float32') / 255.0
y_train_CNN = to_categorical(y_train_CNN)
y_test_CNN = to_categorical(y_test_CNN)
CNN = Sequential()
CNN.add(Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)))
CNN.add(MaxPooling2D((2,2)))
CNN.add(Flatten())
CNN.add(Dense(10, activation='softmax'))
CNN.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
CNN.fit(X_train_CNN, y_train_CNN, epochs=5, batch_size=32)


#Defining the get request for getting the IP address
@app.route('/get_ip', methods=['GET'])
def get_ip():
    return jsonify({'ip': request.remote_addr}), 200

#Defining the get request for getting the results of the Iris dataset
@app.route("/get_iris", methods=['GET'])
def get_iris():
    predictions = randomForest.predict(X_test)
    f1 = f1_score(y_test,predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    return jsonify({
        'f1-score' : f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    })

#Defining the get request for getting the results of the MNIST dataset
@app.route("/get_MNIST", methods=['GET'])
def get_MNIST():
    predictions = CNN.predict_classes(X_test_CNN)
    accuracy = CNN.evaluate(X_test_CNN, y_test_CNN)[1]
    f1 = f1_score(y_test_CNN.argmax(axis=1), predictions)
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test_CNN.argmax(axis=1), predictions, average='macro')
    recall = recall_score(y_test_CNN.argmax(axis=1), predictions, average='macro')
    return jsonify({
        'f1-score': f1,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    })

if __name__ == '__main__':
    app.run(debug=True)
