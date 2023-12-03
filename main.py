import csv
from preprocessing import *
from sklearn import metrics
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


def save_to_csv(rescale, algorithm, x_train, x_test, y_train, y_test, y_pred):


    with open('./data/resultsKNN.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        count_no = y_train['Do Not Drive Advisory'].value_counts()[0] + y_test['Do Not Drive Advisory'].value_counts()[0]
        count_yes = y_train['Do Not Drive Advisory'].value_counts()[1] + y_test['Do Not Drive Advisory'].value_counts()[1]

        row = [rescale,
               algorithm,
               metrics.accuracy_score(y_test, y_pred),
               metrics.precision_score(y_test, y_pred),
               metrics.recall_score(y_test, y_pred),
               metrics.f1_score(y_test, y_pred),
               count_no,
               count_yes,
               str(metrics.confusion_matrix(y_test, y_pred)).replace("\n", "")]

        writer.writerow(row)


def print_stats(y_test, y_pred):
    print("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
    print("Precision: " + str(metrics.precision_score(y_test, y_pred)))
    print("Recall: " + str(metrics.recall_score(y_test, y_pred)))
    print("F1-Score: " + str(metrics.f1_score(y_test, y_pred)))
    print("Confusion Matrix:\n" + str(metrics.confusion_matrix(y_test, y_pred)))


if __name__ == '__main__':
    # specify the path for the dataset
    dataset_path = "./data/Recalls_Data.csv"
    print("Started Program")
    # copy the original dataset and then preprocess the copy
    create_new_copy('./data/Recalls_Data_Original.csv', dataset_path)

    # preprocess the dataset
    x_train, x_test, y_train, y_test = preprocess_csv(dataset_path)


    print("\n\nTraining & Testing KNN Model")
    # Training Model
    knn_model = KNeighborsClassifier(1)
    knn_model.fit(x_train, y_train)
    # Testing Model
    y_pred = knn_model.predict(x_test)
    print_stats(y_test, y_pred)
    # Fewer nearest neighbours makes the algorithm better
    save_to_csv("All KNN", "KNN", x_train, x_test, y_train, y_test, y_pred)
    '''

    print("\n\nTraining & Testing Random Forest Classifier")
    # Training Model
    rf_model = RandomForestClassifier(max_depth=15, n_estimators=20, max_features=1, random_state=42)
    rf_model.fit(x_train, y_train)
    # Testing Model
    y_pred = rf_model.predict(x_test)
    print_stats(y_test, y_pred)
    # Greater max depth makes algorithm better
    # greater n_estimators makes algorithm better
    save_to_csv("A", "Random Forest Classifier", metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), str(metrics.confusion_matrix(y_test, y_pred)))


    print("\n\nTraining & Testing Decision Tree Classifier")
    # Training Model
    rf_model = DecisionTreeClassifier(max_depth=20, random_state=40)
    rf_model.fit(x_train, y_train)
    # Testing Model
    y_pred = rf_model.predict(x_test)
    print_stats(y_test, y_pred)
    # Greater max depth makes algorithm better
    save_to_csv("A", "Decision Tree Classifier", metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), str(metrics.confusion_matrix(y_test, y_pred)))


    print("\n\nTraining & Testing SVM")
    # Training Model
    rf_model = SVC(gamma=2, C=1, random_state=40)
    rf_model.fit(x_train, y_train)
    # Testing Model
    y_pred = rf_model.predict(x_test)
    print_stats(y_test, y_pred)
    save_to_csv("A", "SVM", metrics.accuracy_score(y_test, y_pred), metrics.precision_score(y_test, y_pred), metrics.recall_score(y_test, y_pred), metrics.f1_score(y_test, y_pred), str(metrics.confusion_matrix(y_test, y_pred)))

    '''
