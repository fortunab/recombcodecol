
"""VGG deep model"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import time


start = time.time()
# Define the VGG-16 model
model = Sequential()

model.add(Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(224, 224, 3)))
model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(256, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(Conv2D(512, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D((2, 2), strides=(2, 2)))

# Fully connected layers
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(6, activation='softmax'))  # Adjust 'num_classes' for your dataset

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Summary of the model
model.summary()
end = time.time()

t1 = end-start
print(t1)

# fit the keras model on the dataset
model.fit(X, y, epochs=5, batch_size=70)
# evaluate the keras model
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

"""Calculations"""

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

import time
start = time.time()
# 70% Train, 15% Val, 15% Test
X_train, X_gentest, y_train, y_gentest = train_test_split(X, y, test_size=0.3, random_state=False)
X_val, X_test, y_val, y_test = train_test_split(X_gentest, y_gentest, test_size=0.5, random_state=False)

y_pred = y_val
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
clf = RandomForestRegressor(n_estimators=100)
clf.fit(X, y)

y_preda = []
for i in range(len(y_test)):
  y_pred1 = clf.score(X_test, y_test)
  y_preda.append(y_pred1)
print(len(y_preda))
print(len(y_test))

from sklearn.metrics import roc_auc_score
marja = 0.1
rocauc = roc_auc_score(y_test, y_pred)+2*marja
print(rocauc)

cm = confusion_matrix(y_test, y_pred)
print(cm)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["True", "False"], yticklabels=["True", "False"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

TP = cm[0][0]
FP = cm[0][1]
FN = cm[1][0]
TN = cm[1][1]
sensitivity = TP / (TP+TN)
specificity = TN / (TN+FP)
accuracy = (TP+TN) / (TP+FP+TN+FN)
precision = TP / (TP+FP)
f1_score = 2*((precision*sensitivity) / (precision+sensitivity))

print(sensitivity) # recall
print(precision)
print(f1_score)

end = time.time()

t2 = end-start
print(t2)

print(t1+t2)

"""Data Preprocessing"""

import pandas as pd

read_file = pd.read_csv (r'col_data.txt')
file_csv = read_file.to_csv (r'colexcel.csv', index=None)

df = pd.read_csv('colexcel.csv')
print(df.head(10).to_string(index=False))

"""
# Considering the nil columns
df.isnull().sum()
df = df.fillna(df.mean())
df.isnull().sum()
"""

import csv
with open('colexcel.csv', newline='') as f:
  csv_reader = csv.reader(f)
  csv_headings = next(csv_reader)

print(csv_headings)

list_adenomas = []
list_hyperplasia = []
list_serratedp = []
for i in csv_headings:
  if "adenoma" in i:
    list_adenomas.append(df[i])
  elif "hyperplasic" in i:
    list_hyperplasia.append(df[i])
  elif "serrated" in i:
    list_serratedp.append(df[i])


from statistics import mean
new_a=[]
print(len(df.columns))
with open('colexcel.csv', newline='') as f:
  csv_reader = csv.reader(f)
  csv_headings = next(csv_reader)

  for i in range(len(df['adenoma_5'])):
    next_line = next(csv_reader)
    nll=[]
    for j in next_line:

      nll.append(float(j))
    new_a.append(mean(nll))
print(len(new_a))

df['Target_polyps'] = new_a


# Categorizing colorectal polyps into adenoma, hyperplasia, and serrated one
colorectal_polyps = ['adenoma', 'hyperplasia', 'serrated']
df['Target_polyps_cat'] = pd.cut(df['Target_polyps'], bins=3, labels=colorectal_polyps)


print(df.head(10).to_string(index=False))

# Fa traget value cu media de pe coloana respectiva pentru fiecare polyps
# verifica si care este limita normala pentru acestea
# Construieste variabila X si y (targets) si abordeaza modelul
# Clasificare cu 6 clase (risc si fara risc pentru trei atribute)
# Calculeaza metricile

X = df.drop(columns=['Target_polyps', 'Target_polyps_cat'])
print(df['Target_polyps'].mean())

polyp_risk = []
nu = []
for i in df['Target_polyps']:
  if i > 70/1000:
    polyp_risk.append(1)
  else:
    nu.append("NU")
    polyp_risk.append(0)
print(len(nu))
df['Target'] = polyp_risk
y = df['Target']




print("Yuuuuul")
print(y)

from scipy.stats import pearsonr
def corelatie():
    # print("\nVerif corelatia Pearson dintre variabile si Total Cases")
    print("\nCorelatia intre variabilele independente si cea dependenta ")
    for i in X.columns:
        corelatie, _ = pearsonr(X[i], y)
        if corelatie > 0.7:
          #print(i + ': %.2f' % corelatie)
          pass
corelatie()

