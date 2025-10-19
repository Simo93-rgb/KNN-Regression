import argparse
import os
import time

from sklearn.neighbors import KNeighborsRegressor

from data import fetch_data, edit_dataset
from knn_parallel import KNN_Parallel
from plot import plot_predictions, plot_residuals, plot_comparison
from validazione import KFoldValidation
from valutazione import *

# Percorso del file main.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Percorso alla cartella "Assets" nella directory "KNN Regression"
assets_dir = os.path.join(current_dir, 'Assets')
results_dir = os.path.join(assets_dir, 'results')

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


if __name__ == "__main__":
    # Parser degli argomenti posizionali
    parser = argparse.ArgumentParser(description="KNN with optional standardization and k value.")
    parser.add_argument('--X-standardization', '-X', type=str2bool, nargs='?', default=True,
                        help='Enable/Disable standardization of X (default: True)')
    parser.add_argument('--n-neighbours', '-n', type=int, nargs='?', default=6,
                        help='Value of neighbourhood for KNN (default: 6)')
    parser.add_argument('--test-size', '-t', type=float, nargs='?', default=0.2,
                        help='Value of test-size for KNN (default: 0.2)')
    parser.add_argument('--k-fold', '-k', type=int, nargs='?', default=10,
                        help='Value of k for k-fold cross validation (default: 10)')
    parser.add_argument('--minkowski', '-p', type=int, nargs='?', default=1,
                        help='values of p in the minkowski equation (default: 1)')

    args = parser.parse_args()

    # Utilizzo degli argomenti passati
    X_standardization = args.X_standardization
    n = args.n_neighbours
    test_size = args.test_size
    k_fold = args.k_fold
    minkowski = args.minkowski
    chunk_size = 50
    print(f'Args:\nX_standardization = {X_standardization}\nKNN(k={n})\ntest_size = {test_size}\nk-fold = {k_fold}')

    # Fetch del dataset Combined Cycle Power Plant
    X, y = fetch_data(assets_dir)
    print(f'valore min: {y.min()}\nvalore max: {y.max()}')

    # Manipolazione dataset con opzioni di standardizzazione per X e y
    X_train, X_test, y_train, y_test, x_scaler = edit_dataset(
        X,
        y,
        X_standardization=X_standardization,
        test_size=test_size
    )



    # Validazione incrociata (k-fold) sul set di trai∟ning/validation
    metrix = (
        KFoldValidation(model=KNN_Parallel(k=n, chunk_size=chunk_size, minkowski=minkowski), k_folds=k_fold)
        .validate(X_train, y_train))
    sk_metrix = KFoldValidation(model=KNeighborsRegressor(n_neighbors=n), k_folds=k_fold).validate(X_train, y_train)

    def save_results(metrix, filename=''):
        if os.path.exists(results_dir):
            with open(f'{results_dir}/{filename}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(metrix.keys())  # Scrive le intestazioni
                writer.writerow(metrix.values())  # Scrive i valori
        
            # Salvataggio in un file JSON
            with open(f'{results_dir}/{filename}.json', 'w') as file:
                json.dump(metrix, file, indent=4)
    
    save_results(metrix, 'kfold_knn_parallel')
    save_results(sk_metrix, 'kfold_knn_sklearn')

    print('######## K-FOLD k-NN Parallel ########')
    [print(f"{chiave} su cross validation (k-fold): {valore}") for chiave, valore in metrix.items()]

    print('######## K-FOLD k-NN sklearn ########')
    [print(f"{chiave} su cross validation (k-fold): {valore}") for chiave, valore in sk_metrix.items()]

    # Creazione del modello KNN
    knn = KNN_Parallel(k=n, minkowski=minkowski, chunk_size=chunk_size)
    knn_regressor = KNeighborsRegressor(n_neighbors=n)

    # Addestramento finale sul training set completo
    start_time = time.time()
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)
    end_time = time.time()


    start_time_sk = time.time()
    knn_regressor.fit(X_train, y_train)
    y_test_pred_sk = knn_regressor.predict(X_test)
    end_time_sk = time.time()

    # Se è stata applicata la standardizzazione su X, esegui l'inverso della trasformazione
    if x_scaler:
        X_train_rescaled = x_scaler.inverse_transform(X_train)
        X_test_rescaled = x_scaler.inverse_transform(X_test)
    else:
        X_train_rescaled = X_train
        X_test_rescaled = X_test

    # Valutazione del modello
    print('######## Evaluating my model ########')
    test_metrix = evaluate_model(y_true=y_test, y_pred=y_test_pred, message="Test Set", savefile=True)

    print('######## Evaluating sklearn ########')
    test_metrix_sk = evaluate_model(y_true=y_test, y_pred=y_test_pred_sk, message="Test Set", savefile=True)

    save_results(test_metrix, 'test_knn_parallel')
    save_results(test_metrix_sk, 'test_knn_sklearn')

    # Visualizzazioni
    # Plot della curva di apprendimento

    plot_predictions(y_test, y_test_pred, model_name="KNN Parallel", assets_dir=assets_dir)
    plot_predictions(y_test, y_test_pred_sk, model_name="KNN sklearn", assets_dir=assets_dir)
    plot_residuals(y_test, y_test_pred, model_name="KNN Parallel", assets_dir=assets_dir)
    plot_residuals(y_test, y_test_pred_sk, model_name="KNN sklearn", assets_dir=assets_dir)
    plot_comparison(metrix, sk_metrix, title='Cross Validation')
    plot_comparison(test_metrix, test_metrix_sk, title='Test')

    # Stampa dei tempi di esecuzione
    print(f'Tempo esecuzione k-NN Parallel: {end_time - start_time}')
    print(f'Tempo esecuzione k-NN sklearn: {end_time_sk - start_time_sk}')