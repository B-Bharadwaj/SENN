import matplotlib.pyplot as plt
import pandas as pd

def plot_metrics():
    # Read data from metrics.csv
    df = pd.read_csv('metrics.csv')

    # Accuracy vs Params
    plt.figure()
    plt.scatter(df['param_count'], df['val_accuracy'])
    plt.title('Accuracy vs Params')
    plt.xlabel('Number of Parameters')
    plt.ylabel('Validation Accuracy')
    plt.savefig('outputs/accuracy_vs_params.png')

    # Accuracy vs FLOPs
    plt.figure()
    plt.scatter(df['flops'], df['val_accuracy'])
    plt.title('Accuracy vs FLOPs')
    plt.xlabel('FLOPs')
    plt.ylabel('Validation Accuracy')
    plt.savefig('outputs/accuracy_vs_flops.png')

    # Accuracy vs Latency (Optional)
    if 'latency' in df.columns:
        plt.figure()
        plt.scatter(df['latency'], df['val_accuracy'])
        plt.title('Accuracy vs Latency')
        plt.xlabel('Latency (seconds)')
        plt.ylabel('Validation Accuracy')
        plt.savefig('outputs/accuracy_vs_latency.png')

# Call the plot function to generate and save the plots
plot_metrics()
