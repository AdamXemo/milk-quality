# Libs
import pandas as pd
from keras.models import Sequential
from keras.layers import Flatten, Dense
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Reading data
data = pd.read_csv('/home/adam/Datasets/milknew.csv')

# Converting low,medium,high to 0,1,2
def grade_to_number(grades):
    y = []
    for grade in grades:
        if grade == 'low':
            grade = 0
        elif grade == 'medium':
            grade = 1
        else:
            grade = 2
        y.append(grade)
    return y

# Assigning X,y
grades = data['Grade']
y = grade_to_number(grades)
X = data.drop('Grade', axis=1)

# Spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Scaling
sc = StandardScaler()
# Transforming training data
X_train = sc.fit_transform(X_train)
# And testing data aswell
X_test = sc.fit_transform(X_test)

# To np.array
X_train = np.array(X_train)
y_train = np.array(y_train)
X_test = np.array(X_test)
y_test = np.array(y_test)

# Creating model
model = Sequential()

# Layers
model.add(Flatten(input_shape=X_train[0].shape))
model.add(Dense(14, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(28, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(28, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(14, activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# Compiling, fitting, testing, predicting
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, epochs=20, batch_size=1)
model.evaluate(X_test, y_test, batch_size=1)
predictions = model.predict(X_test)

# Saving
model.save('model.h5')


