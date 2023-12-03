import os
import csv
from preprocessing import *
from metrics_graph import *
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.neighbors import KNeighborsClassifier


def save_to_csv(rescale, algorithm, x_train, x_test, y_train, y_test, y_pred):

    with open('./data/resultsKNN2.csv', 'a', newline='') as file:
        writer = csv.writer(file)

        file_size = os.path.getsize('./data/resultsKNN2.csv')
        if file_size == 0:
            header = ['Rescale Algorithm', 'ML Algorithm', 'Accuracy', 'Precision', 'Recall', 'F1', 'No', 'Yes', 'Confusion Matrix']
            writer.writerow(header)

        rescale_map = {'nm': 'Near Miss',
                       'cc': 'Cluster Centroids',
                       'rus(not minority)': 'Random Under Sampler (Not Minority)',
                       'ros(0.1)rus(0.5)': 'OverSampler(0.2) then UnderSampler(0.5)',
                       'smoteenn': 'SMOTEENN',
                       'none': 'None - Original Data'}

        count_no = y_train['Do Not Drive Advisory'].value_counts()[0] + y_test['Do Not Drive Advisory'].value_counts()[0]
        count_yes = y_train['Do Not Drive Advisory'].value_counts()[1] + y_test['Do Not Drive Advisory'].value_counts()[1]

        row = [rescale_map[rescale],
               algorithm,
               metrics.accuracy_score(y_test, y_pred),
               metrics.precision_score(y_test, y_pred),
               metrics.recall_score(y_test, y_pred),
               metrics.f1_score(y_test, y_pred),
               count_no,
               count_yes,
               metrics.confusion_matrix(y_test, y_pred).tolist()]
        writer.writerow(row)


def print_stats(y_test, y_pred):
    print("Accuracy: " + str(metrics.accuracy_score(y_test, y_pred)))
    print("Precision: " + str(metrics.precision_score(y_test, y_pred)))
    print("Recall: " + str(metrics.recall_score(y_test, y_pred)))
    print("F1-Score: " + str(metrics.f1_score(y_test, y_pred)))
    print("Confusion Matrix:\n" + str(metrics.confusion_matrix(y_test, y_pred)))
    print(type(metrics.confusion_matrix(y_test, y_pred)))
    print(metrics.confusion_matrix(y_test, y_pred).shape)

def train_test_knn(dataset_path, option, n):
    # copy the original dataset and then preprocess the copy
    create_new_copy('./data/Recalls_Data_Original.csv', dataset_path)

    # preprocess the dataset
    x_train, x_test, y_train, y_test = preprocess_csv(dataset_path, option)

    print("\n\nTraining & Testing KNN Model")
    # Training Model
    knn_model = KNeighborsClassifier(n)
    knn_model.fit(x_train, y_train)
    # Testing Model
    y_pred = knn_model.predict(x_test)
    # Plot ROC Curve
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    plt.plot(fpr, tpr, label="KNN " + option + ", AUC=" + str(auc))
    print_stats(y_test, y_pred)
    # Fewer nearest neighbours makes the algorithm better
    save_to_csv(option, "KNN", x_train, x_test, y_train, y_test, y_pred)

if __name__ == '__main__':
    # settings for plotting ROC Curve
    X, y = load_wine(return_X_y=True)
    y = y == 2

    # specify the path for the dataset
    dataset_path = "./data/Recalls_Data.csv"
    print("Started Program")

    train_test_knn(dataset_path, 'none', 5)
    train_test_knn(dataset_path, 'nm', 5)
    train_test_knn(dataset_path, 'cc', 5)
    train_test_knn(dataset_path, 'rus(not minority)', 5)
    train_test_knn(dataset_path, 'ros(0.1)rus(0.5)', 5)
    train_test_knn(dataset_path, 'smoteenn', 5)

    plt.legend()
    plt.show()

    plot_metrics('./data/resultsKNN2.csv')
    confusion_plot('./data/resultsKNN2.csv')

