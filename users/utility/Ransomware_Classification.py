import os

import numpy as np
import pandas as pd
from django.conf import settings
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

# from mlxtend.plotting import plot_confusion_matrix

path = os.path.join(settings.MEDIA_ROOT, "dataset", 'Ransomware.csv')
df = pd.read_csv(path, sep='|')

# plt.pie(df.legitimate.value_counts().values.tolist(), labels=['Safe','Ransomware'], autopct='%.2f%%')
# plt.legend()
# plt.show()
# sns.heatmap(df.corr())
print("Ramram Raja:", df.legitimate.value_counts())


def vifScore():
    # Using VIF to remove highly correlated columns
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    cols_vif = df.columns.tolist()
    cols_vif.remove('legitimate')
    cols_vif.remove('md5')
    cols_vif.remove('Name')
    cols_vif

    # VIF dataframe
    vif_data = pd.DataFrame()
    vif_data["feature"] = cols_vif

    # calculating VIF for each feature
    vif_data["VIF"] = [variance_inflation_factor(df[cols_vif].values, i)
                       for i in range(len(cols_vif))]

    print(vif_data)
    return vif_data.to_html(index=False)


df.drop(
    ['MinorImageVersion', 'MinorSubsystemVersion', 'SizeOfHeapCommit', 'SectionsMinRawsize', 'SectionsMinVirtualsize',
     'SectionMaxVirtualsize'], axis=1, inplace=True)


def iv_woe(data, target, bins=10, show_woe=False):
    # Empty Dataframe
    newDF, woeDF = pd.DataFrame(), pd.DataFrame()

    # Extract Column Names
    cols = data.columns

    # Run WOE and IV on all the independent variables
    for ivars in cols[~cols.isin([target])]:
        if (data[ivars].dtype.kind in 'bifc') and (len(np.unique(data[ivars])) > 10):
            binned_x = pd.qcut(data[ivars], bins, duplicates='drop')
            d0 = pd.DataFrame({'x': binned_x, 'y': data[target]})
        else:
            d0 = pd.DataFrame({'x': data[ivars], 'y': data[target]})
        d = d0.groupby("x", as_index=False).agg({"y": ["count", "sum"]})
        d.columns = ['Cutoff', 'N', 'Events']
        d['% of Events'] = np.maximum(d['Events'], 0.5) / d['Events'].sum()
        d['Non-Events'] = d['N'] - d['Events']
        d['% of Non-Events'] = np.maximum(d['Non-Events'], 0.5) / d['Non-Events'].sum()
        d['WoE'] = np.log(d['% of Events'] / d['% of Non-Events'])
        d['IV'] = d['WoE'] * (d['% of Events'] - d['% of Non-Events'])
        d.insert(loc=0, column='Variable', value=ivars)
        print("Information value of " + ivars + " is " + str(round(d['IV'].sum(), 6)))
        temp = pd.DataFrame({"Variable": [ivars], "IV": [d['IV'].sum()]}, columns=["Variable", "IV"])
        newDF = pd.concat([newDF, temp], axis=0)
        woeDF = pd.concat([woeDF, d], axis=0)

        # Show WOE Table
        if show_woe == True:
            print(d)
    return newDF, woeDF


df.legitimate = df.legitimate.astype('int64')
iv, woe = iv_woe(df.drop(['Name'], axis=1), 'legitimate')
iv.sort_values(by='IV', ascending=False)
features = iv.sort_values(by='IV', ascending=False)['Variable'][:15].values.tolist()

X = df[features]
y = df['legitimate']

randomseed = 42
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
print(X_test.shape[0] + X_train.shape[0])
print('Training labels shape:', y_train.shape)
print('Test labels shape:', y_test.shape)
print('Training features shape:', X_train.shape)
print('Test features shape:', X_test.shape)

from collections import Counter
import imblearn

"""
Before SMOTE_Tomek
"""
counter_train = Counter(y_train)
counter_test = Counter(y_test)
print(counter_train, counter_test)

# creating imblearn resampling object
# sampling strategy is the propotion of output
# resampled data that is the minority class
over_and_under_sample = imblearn.combine.SMOTETomek(sampling_strategy=1.0, n_jobs=-1, random_state=randomseed)
X_train, y_train = over_and_under_sample.fit_resample(X_train, y_train)

# checking under- and over-sample ratios between train and test set.
# DO NOT resample the test set!
counter_train = Counter(y_train)
counter_test = Counter(y_test)
print(counter_train, counter_test)

print(X_test.shape[0] + X_train.shape[0])
print('Training labels shape:', y_train.shape)
print('Test labels shape:', y_test.shape)
print('Training features shape:', X_train.shape)
print('Test features shape:', X_test.shape)


def process_decisionTree():
    # 1 Decision Tree
    from sklearn.tree import DecisionTreeClassifier
    dt = DecisionTreeClassifier()
    # Fitting training set to the model
    dt.fit(X_train, y_train)
    # Predicting the test set results based on the model
    y_pred = dt.predict(X_test)
    # Calculate the accuracy score of this model
    dt_acc = accuracy_score(y_test, y_pred)
    dt_report = classification_report(y_test, y_pred, output_dict=True)
    print('Accuracy of DT model is ', dt_acc)
    from sklearn.metrics import roc_auc_score
    auc = np.round(roc_auc_score(y_test, y_pred), 3)
    print("Auc for DT sample data is {}".format(auc))
    return dt_acc, dt_report


def process_randomForest():
    # 3. Random Forest Classifier
    RFC_model = RandomForestClassifier(random_state=randomseed)
    # Fitting training set to the model
    RFC_model.fit(X_train, y_train)
    # Predicting the test set results based on the model
    rfc_y_pred = RFC_model.predict(X_test)
    # Calculate the accuracy score of this model
    rf_acc = accuracy_score(y_test, rfc_y_pred)
    rf_report = classification_report(y_test, rfc_y_pred, output_dict=True)
    print('Accuracy of RFC model is ', rf_acc)
    from sklearn.metrics import roc_auc_score
    auc = np.round(roc_auc_score(y_test, rfc_y_pred), 3)
    print("Auc for RF sample data is {}".format(auc))
    return rf_acc, rf_report


def process_naiveBayes():
    # 3. Naive Bayes
    nb_model = GaussianNB()
    # Fitting training set to the model
    nb_model.fit(X_train, y_train)
    # Predicting the test set results based on the model
    y_pred = nb_model.predict(X_test)
    # Calculate the accuracy score of this model
    nb_acc = accuracy_score(y_test, y_pred)
    nb_report = classification_report(y_test, y_pred, output_dict=True)
    print('Accuracy of NB model is ', nb_acc)
    from sklearn.metrics import roc_auc_score
    auc = np.round(roc_auc_score(y_test, y_pred), 3)
    print("Auc for NB sample data is {}".format(auc))
    return nb_acc, nb_report


def process_logisticRegression():
    # 4. Logistic Regression
    lg_model = LogisticRegression()
    # Fitting training set to the model
    lg_model.fit(X_train, y_train)
    # Predicting the test set results based on the model
    y_pred = lg_model.predict(X_test)
    # Calculate the accuracy score of this model
    lg_acc = accuracy_score(y_test, y_pred)
    lg_report = classification_report(y_test, y_pred, output_dict=True)
    print('Accuracy of Logistic Regression model is ', lg_acc)
    from sklearn.metrics import roc_auc_score
    auc = np.round(roc_auc_score(y_test, y_pred), 3)
    print("Auc for Logistic Regression sample data is {}".format(auc))
    return lg_acc, lg_report


def process_neuralNetwork():
    # 5. Neural Network
    nn_model = Perceptron()
    # Fitting training set to the model
    nn_model.fit(X_train, y_train)
    # Predicting the test set results based on the model
    y_pred = nn_model.predict(X_test)
    # Calculate the accuracy score of this model
    nn_acc = accuracy_score(y_test, y_pred)
    nn_report = classification_report(y_test, y_pred, output_dict=True)
    print('Accuracy of Neural Network model is ', nn_acc)
    from sklearn.metrics import roc_auc_score
    auc = np.round(roc_auc_score(y_test, y_pred), 3)
    print("Auc for Neural Network sample data is {}".format(auc))
    return nn_acc, nn_report


def user_prediction():
    model = RandomForestClassifier(random_state=randomseed)
    # model = GaussianNB()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    X_test['actual_values'] = y_test
    X_test['prediction_values'] = y_pred
    print(f"X_test {type(X_test)}, Y_Test {type(y_test)} y_pred {type(y_pred)}")
    X_test['actual_values'] = X_test['actual_values'].map({0: 'Safe', 1: 'Attacked'})
    X_test['prediction_values'] = X_test['prediction_values'].map({0: 'Safe', 1: 'Attacked'})
    data = X_test[X_test.columns[::-1]]
    return data.to_html(index=False)
