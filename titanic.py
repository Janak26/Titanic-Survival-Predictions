import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Data
train_df = pd.read_csv("C:/Users/Admin/Desktop/Titanic project/train.csv")
test_df = pd.read_csv("C:/Users/Admin/Desktop/Titanic project/test.csv")

train_data = train_df.copy()
test_data = test_df.copy()

# Visualizing Survived people wrt class
# survived_by_class = train_df.groupby('Pclass')['Survived'].sum()
sns.barplot('Pclass', 'Survived', data=train_data, color='turquoise')
# plt.show()


# Fill missing age with median
train_data["Age"].fillna(train_df["Age"].median(skipna=True), inplace=True)

# Visualize age wrt survived
plt.figure(figsize=(15,8))
ax = sns.kdeplot(train_data["Age"][train_data.Survived == 1], color='darkturquoise', shade=True)
sns.kdeplot(train_data['Age'][train_data.Survived == 0], color='coral', shade=True)
plt.legend(['Survived', 'Died'])
plt.title('Density plot of Age for Surviving and Deceased')
ax.set(xlabel='Age')
plt.xlim(-10, 100)
# plt.show()






# Create a child coloumn
train_data["Child"] = (train_data['Age']<=16).astype(int)
test_data["Child"] = (test_data['Age']<=16).astype(int)


# Remove Ticket and Cabin
train_data.drop(['Ticket', 'Cabin'], axis=1, inplace=True)
test_data.drop(['Ticket', 'Cabin'], axis=1, inplace=True)

# Plot survived classes
survived_1 = train_data[train_data['Pclass']==1]['Survived'].value_counts()
survived_2 = train_data[train_data['Pclass']==2]['Survived'].value_counts()
survived_3 = train_data[train_data['Pclass']==3]['Survived'].value_counts()
df = pd.DataFrame([survived_1,survived_2,survived_3])
df['total']=df[0]+df[1]
df.index = ['1st class','2nd class','3rd class']
df.rename(index=str,columns={0:'Survived',1:'Died'})
print (df)
df.plot(kind='bar',label=['Survived','Died'])
# plt.show()


# Identify place for maximum embarkment
sns.countplot(x='Embarked', data=train_data)
sns.countplot(x='Embarked', data=test_data)
# plt.show()

# fill empty Embarked with Max number of Embarkments
train_data['Embarked'].fillna(train_df['Embarked'].value_counts().idxmax(), inplace=True)
test_data['Embarked'].fillna(test_df['Embarked'].value_counts().idxmax(), inplace=True)

# Convert embarkments to integers
for df in train_data, test_data:
    df["Embarked"] = df["Embarked"].map(dict(zip(("S", "C", "Q"), (0,1,2))))



# Separate Pclass to Pclass_1, Pclass_2, Pclass_3, Sex to Sex_female, Sex_male
train_data = pd.get_dummies(train_data, columns=["Pclass", "Sex"])
test_data = pd.get_dummies(test_data, columns=["Pclass", "Sex"])
# print(test_data.head())

# Visualizing survived and dead wrt fares
survived = train_data[train_data["Survived"] == 1]["Fare"]
died = train_data[train_data["Survived"] == 0]["Fare"]

fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
fig.set_size_inches(12,8)
fig.subplots_adjust(hspace = 0.5)
ax1.hist(survived, facecolor = 'darkgreen', alpha = 0.75)
ax1.set(title="Survived wrt fare", xlabel="Fare", ylabel="Numbers")
ax2.hist(died, facecolor = 'darkred', alpha = 0.75)
ax2.set(title="Died wrt fare", xlabel="Fare", ylabel="Numbers")
plt.show()


# Finding family size of individual persons
for df in train_data, test_data:
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
plt.show()

# Prediction Random Forests
from sklearn.model_selection import train_test_split

predictors = ["Age", "Embarked", "Child", "Pclass_1", "Pclass_2", "Pclass_3", "Sex_female", "FamilySize"]
X_train, X_test, y_train, y_test = train_test_split(train_data[predictors], train_data["Survived"])


from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=10, min_samples_leaf=5, random_state=0)
forest.fit(X_train, y_train)
print("Random forest score: {0:.2}".format(forest.score(X_test, y_test)))



# Predict using Logistic regression
from sklearn.linear_model import LogisticRegression
data_set = train_data[["Age", "Embarked", "Child", "Pclass_1", "Pclass_2", "Pclass_3", "Sex_female", "FamilySize"]]
training_predictors = pd.get_dummies(data_set)
X1 = training_predictors
y = train_data["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X1, y, test_size=0.3, random_state=1)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print("Accuracy of logistic regression: {:.2f}".format(logreg.score(X_test, y_test)))

















