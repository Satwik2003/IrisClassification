import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#importing csv
df = pd.read_csv('iris.csv')

#processing data
df = df.drop_duplicates()

#making target column
df["Target"] = 0
conditions = [df["Species"] == "Iris-versicolor", df["Species"] == "Iris-virginica"]
choices = [1, 2]
df["Target"] = np.select(conditions, choices, default=df["Target"])



#Making test train splits
data = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
labels = df["Target"]

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size= 0.4)

#Classifier Algo
knn = KNeighborsClassifier(n_neighbors=12)
knn.fit(x_train, y_train)


#get inputs

inp = []

inp.append(float(input("Enter Sepal Length in cm:\t")))
inp.append(float(input("Enter Sepal Width in cm:\t")))
inp.append(float(input("Enter Petal Length in cm:\t")))
inp.append(float(input("Enter Petal Width in cm:\t")))

inp = np.array(inp).reshape(1,4)

df_test = pd.DataFrame(inp,index=[0], columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

#make prediciton

pred = knn.predict(df_test)

if pred[0] == 0:
    print("\n\n\nIt is Setosa.\n\n\n")

elif pred[0] == 1:
    print("\n\n\nIt is Versicolor.\n\n\n")

elif pred[0] == 2:
    print("\n\n\nIt is Virginica.\n\n\n")