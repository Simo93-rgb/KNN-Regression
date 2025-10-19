import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import root_mean_squared_error, explained_variance_score, mean_absolute_error
from valutazione import validate_predictions
import seaborn as sns
from data import fetch_data, edit_dataset
from knn_parallel import  KNN_Parallel
import os
import argparse
from valutazione import explained_variance, mean_squared_error

# Percorso del file main.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Percorso alla cartella "Assets"
assets_dir = os.path.join(current_dir, 'Assets')

# Percorso alla cartella "results"
results_dir = os.path.join(assets_dir, 'results')

def plot_predictions(y_true, y_pred, model_name="", assets_dir=""):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    x_dim, y_dim = [16, 9]
    plt.figure(figsize=(x_dim, y_dim))
    plt.scatter(y_true, y_pred, alpha=0.5, color='green')
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color='red', lw=2)
    plt.title(f'Confronto Valori Predetti e Reali - {model_name}', fontsize=24)
    plt.xlabel('Valori Reali', fontsize=18)
    plt.ylabel('Valori Predetti', fontsize=18)
    plt.savefig(f'{results_dir}/predictions_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_residuals(y_true, y_pred, model_name="", assets_dir=""):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    x_dim, y_dim = [16, 9]
    residuals = y_true - y_pred
    plt.figure(figsize=(x_dim, y_dim))
    plt.hist(residuals, bins=60, color='green')
    plt.title(f'Distribuzione dei Residui - {model_name}', fontsize=24)
    plt.xlabel('Errore di Predizione', fontsize=18)
    plt.ylabel('Frequenza', fontsize=18)
    plt.savefig(f'{results_dir}/residuals_{model_name}.png', format='png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_corr_matrix(X, assets_dir=""):
    x_dim, y_dim = [20, 14]
    corr_matrix = np.corrcoef(X, rowvar=False)
    plt.figure(figsize=(x_dim, y_dim))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title("Matrice di Correlazione delle Feature", fontsize=24)
    plt.savefig(f'{assets_dir}/correlation_matrix.png', format='png', dpi=600, bbox_inches='tight')
    # plt.show()
    plt.close()


def plot_learning_curve(model, X_train, y_train, X_test, y_test, assets_dir="", file_name='learning_curve', x_scaler=None):
    print('Plotting Learning Curve')
    """ Plotta la curva di apprendimento per un dato modello KNN. """
    y_train = y_train.values
    y_test = y_test.values
    train_errors = []
    val_errors = []

    # Dividiamo il dataset di training in dimensioni crescenti
    training_sizes = np.linspace(0.1, 1.0, 10)  # dal 10% al 100% del training set

    for size in training_sizes:
        # Determina il numero di campioni da utilizzare
        current_size = int(size * len(X_train))
        print(f'Training on {current_size} samples')  # Aggiungi questa riga per verificare le dimensioni
        
        # Prendi un sottoinsieme di X_train e y_train
        X_train_subset = X_train[:current_size]
        y_train_subset = y_train[:current_size]

        # Riaddestra il modello sul sottoinsieme corrente
        model.fit(X_train_subset, y_train_subset)

        # Calcola l'errore sul training set (sottoinsieme corrente)
        y_train_pred = model.predict(X_train_subset)
        train_mse = root_mean_squared_error(y_train_subset, y_train_pred)  # RMSE per il subset corrente
        train_errors.append(train_mse)

        # Calcola l'errore sul validation set (intero test set)
        y_val_pred = model.predict(X_test)
        val_mse = root_mean_squared_error(y_test, y_val_pred)
        val_errors.append(val_mse)

        print(f'Training RMSE: {train_mse}, Validation RMSE: {val_mse}')  # Aggiungi per debug

        # Se è stata applicata la standardizzazione su X, esegui l'inverso della trasformazione


    # Plot della learning curve
    plt.figure(figsize=(16, 9))
    plt.plot(training_sizes * len(X_train), train_errors, label='Errore Training Set', marker='o')
    plt.plot(training_sizes * len(X_train), val_errors, label='Errore Validation Set', marker='o')
    plt.xlabel('Dimensione Training Set')
    plt.ylabel('RMSE')
    plt.title('Curva di Apprendimento - KNN')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'{assets_dir}/{file_name}.png', format='png', dpi=600, bbox_inches='tight')

    # plt.show()

def plot_rmse_vs_n_neighbors(
        X_train: np.ndarray, 
        y_train: np.ndarray, 
        X_test: np.ndarray, 
        y_test: np.ndarray, 
        n_neighbors_range: range = range(1, 51)
    ) -> None:
    """
    Plotta l'RMSE al variare del numero di vicini più vicini (n_neighbors) nel KNN, 
    sia per il train che per il test.

    Parameters:
    - X_train (np.ndarray): Le feature del set di addestramento.
    - y_train (np.ndarray): I target del set di addestramento.
    - X_test (np.ndarray): Le feature del set di test.
    - y_test (np.ndarray): I target del set di test.
    - n_neighbors_range (range): Un range di valori di n_neighbors da testare (default: range(1, 51)).

    Returns:
    - None: La funzione plotta il grafico ma non ritorna alcun valore.
    """
    # Liste per memorizzare i valori di RMSE
    rmse_train_list = []
    rmse_test_list = []

    # Loop attraverso i valori di n_neighbors
    for k in n_neighbors_range:
        knn = KNN_Parallel(n_neighbors=k)
        knn.fit(X_train, y_train)
        
        # Predizione sul train
        y_train_pred = knn.predict(X_train)
        rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
        rmse_train_list.append(rmse_train)
        
        # Predizione sul test
        y_test_pred = knn.predict(X_test)
        rmse_test = np.sqrt(mean_squared_error(y_test, y_test_pred))
        rmse_test_list.append(rmse_test)

    # Plot dei risultati
    plt.figure(figsize=(10, 6))
    plt.plot(n_neighbors_range, rmse_train_list, label='Train RMSE', marker='o')
    plt.plot(n_neighbors_range, rmse_test_list, label='Test RMSE', marker='o')
    plt.xlabel('Number of Neighbors (k)')
    plt.ylabel('RMSE')
    plt.title('RMSE vs Number of Neighbors in KNN')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    # Funzione per convertire stringhe in booleani
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
    
    # Parser degli argomenti posizionali
    parser = argparse.ArgumentParser(description="KNN with optional standardization and k value.")
    parser.add_argument('--X_standardization', '-X', type=str2bool, nargs='?', default=True, help='Enable/Disable standardization of X (default: True)')
    parser.add_argument('--y_standardization', '-y', type=str2bool, nargs='?', default=True, help='Enable/Disable standardization of y (default: True)')
    parser.add_argument('--n_neighboors', '-n', type=int, nargs='?', default=50, help='Value of k for KNN (default: 50)')
    parser.add_argument('--test_size', '-t', type=float, nargs='?', default=0.02, help='Value of k for KNN (default: 0.2)')

    args = parser.parse_args()

    # Utilizzo degli argomenti passati
    X_standardization = args.X_standardization
    y_standardization = args.y_standardization
    k = args.k
    test_size=args.test_size

    # Carica i dati e prepara il dataset
    X, y = fetch_data(assets_dir)
    X_train, X_test, y_train, y_test, x_scaler = edit_dataset(X, y, X_standardization=X_standardization, test_size=test_size)

    # Crea il modello KNN
    knn = KNN_Parallel(k=k)
    print(f'Parallel Plotting:\nX_standardization = {X_standardization}\ny_standardization = {y_standardization}\nKNN(k={k})\ntest_size = {test_size}')

    # Plot della learning curve
    plot_learning_curve(
        knn, 
        X_train, 
        y_train, 
        X_test, 
        y_test, 
        assets_dir, 
        file_name=f'/plot/learning_curve_X_st_{X_standardization}_y_st_{y_standardization}_k_{k}_test_size_{test_size}',
        x_scaler=x_scaler
    )

def plot_comparison(knn_metrics:dict, knn_sklearn_metrics:dict, title:str):
    """
    Plots a comparison histogram between KNN and KNN_Sklearn metrics.

    Parameters:
    knn_metrics (dict): Dictionary containing KNN metrics.
    knn_sklearn_metrics (dict): Dictionary containing KNN_Sklearn metrics.
    title (str): Title of the plot.
    """
    # Extract metric names and values
    metrics = list(knn_metrics.keys())
    knn_values = list(knn_metrics.values())
    knn_sklearn_values = list(knn_sklearn_metrics.values())

    # Set the position of the bars on the x-axis
    x = range(len(metrics))

    # Plotting the bars
    fig, ax = plt.subplots()
    bar_width = 0.35
    ax.bar(x, knn_values, width=bar_width, label='KNN')
    ax.bar([p + bar_width for p in x], knn_sklearn_values, width=bar_width, label='KNN_Sklearn')

    # Adding labels and title
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title(title)
    ax.set_xticks([p + bar_width / 2 for p in x])
    ax.set_xticklabels(metrics)
    ax.legend()

    # Show the plot
    plt.savefig(f'{results_dir}/comparison_{title}.png', format='png', dpi=600, bbox_inches='tight')
    plt.show()