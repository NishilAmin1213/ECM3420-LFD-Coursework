import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

def confusion_plot(csvfile='./data/resultsKNN1.csv'):
    print("Generating Confusion Matrices....")
    metrics = pd.read_csv(csvfile)
    algorithms = metrics['Rescale Algorithm'].values
    confusion_arrays = metrics['Confusion Matrix'].values

    for i in range(0, len(algorithms)):
        string_array = confusion_arrays[i]
        matrix = ast.literal_eval(string_array)
        cm = np.asarray(matrix)

        cm_plot = ConfusionMatrixDisplay(cm)
        cm_plot.plot(cmap=plt.get_cmap('Blues'))

        plt.title('Confusion Matrix for ' + algorithms[i])
        plt.show()

def plot_metrics(csvfile='./data/resultsKNN1.csv'):
    print("Generating metrics graph ....")
    metrics = pd.read_csv(csvfile)

    algorithms = metrics['Rescale Algorithm'].values
    accuracy = metrics['Accuracy'].values
    precision = metrics['Precision'].values
    recall = metrics['Recall'].values
    f1 = metrics['F1'].values

    y_scale = np.arange(0, 1.2, 0.2)

    # set up plot
    plt.figure(figsize=(8, 4.5))
    plt.grid(True)
    plt.title('Performance Metrics for different Resampling Algorithms')

    # plot each line
    for i in range(0, len(algorithms)):
        plt.plot([accuracy[i], precision[i], recall[i], f1[i]], label=algorithms[i], marker='o')

    # defein x-axis labels
    plt.xticks(np.arange(4), ['Accuracy', 'Precision', 'Recall', 'F1 Score'])
    plt.xlabel('Metrics')

    # define y axis labels
    plt.yticks(y_scale)
    plt.ylabel('Value')

    # define parameters for the plot as a whole
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()

if __name__ == '__main__':
    plot_metrics()
    confusion_plot()
