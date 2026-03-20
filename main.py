import argparse
import csv
import json
import os
import time

from sklearn.neighbors import KNeighborsRegressor

from src.data import fetch_data, edit_dataset
from src.knn_parallel import KNN_Parallel
from src.plot import plot_predictions, plot_residuals, plot_comparison
from src.validazione import KFoldValidation
from src.valutazione import evaluate_model

# Percorso del file main.py
current_dir = os.path.dirname(os.path.abspath(__file__))

# Percorso alla cartella "assets" nella directory "KNN Regression"
assets_dir = os.path.join(current_dir, 'assets')
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
    os.makedirs(results_dir, exist_ok=True)

    # Parser degli argomenti posizionali
    parser = argparse.ArgumentParser(
        description="KNN regression custom vs scikit-learn on CCPP dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--X-standardization', '-X', type=str2bool, nargs='?', default=True,
                        help='Enable/Disable standardization of X')
    parser.add_argument('--n-neighbours', '-n', type=int, nargs='?', default=6,
                        help='Value of neighbourhood for KNN')
    parser.add_argument('--test-size', '-t', type=float, nargs='?', default=0.2,
                        help='Value of test-size for KNN')
    parser.add_argument('--k-fold', '-k', type=int, nargs='?', default=10,
                        help='Value of k for k-fold cross validation')
    parser.add_argument('--minkowski', '-p', type=int, nargs='?', default=1,
                        help='Values of p in the Minkowski distance')
    parser.add_argument('--auto-tune', action='store_true',
                        help='Automatically tune k and Minkowski p for the custom model using CV')
    parser.add_argument('--help-args', action='store_true',
                        help='Show detailed argument guide and exit')

    args = parser.parse_args()

    if args.help_args:
        parser.print_help()
        print(
            "\nGuida rapida:\n"
            "  -X / --X-standardization : standardizza le feature prima del training\n"
            "  -n / --n-neighbours      : numero di vicini del KNN\n"
            "  -t / --test-size         : quota del dataset riservata al test (0-1)\n"
            "  -k / --k-fold            : numero di fold per cross validation\n"
            "  -p / --minkowski         : ordine p della distanza di Minkowski (1=Manhattan, 2=Euclidea)\n"
            "  --auto-tune              : ricerca automatica dei migliori valori di k e p (solo modello custom)\n"
        )
        raise SystemExit(0)

    # Utilizzo degli argomenti passati
    X_standardization = args.X_standardization
    n = args.n_neighbours
    test_size = args.test_size
    k_fold = args.k_fold
    minkowski = args.minkowski
    auto_tune = args.auto_tune
    chunk_size = 50
    print(
        f'Args:\nX_standardization = {X_standardization}\nKNN(k={n}, p={minkowski})\n'
        f'test_size = {test_size}\nk-fold = {k_fold}\nauto_tune = {auto_tune}'
    )

    # Fetch del dataset Combined Cycle Power Plant
    X, y = fetch_data(assets_dir)
    print(f'valore min: {y.min()}\nvalore max: {y.max()}')

    # Split train/test raw per CV leak-safe.
    X_train_raw, _, y_train_raw, _, _ = edit_dataset(
        X,
        y,
        X_standardization=False,
        test_size=test_size,
    )

    # Split train/test per il training finale.
    X_train, X_test, y_train, y_test, _ = edit_dataset(
        X,
        y,
        X_standardization=X_standardization,
        test_size=test_size,
        assets_dir=assets_dir,
    )



    # Validazione incrociata leak-safe: scaler rifittato ad ogni fold.
    custom_model_for_cv = KNN_Parallel(k=n, chunk_size=chunk_size, minkowski=minkowski)
    custom_validator = KFoldValidation(
        model=custom_model_for_cv,
        k_folds=k_fold,
        standardize_X=X_standardization,
    )

    if auto_tune:
        metrix = custom_validator.validate_and_find_best_hyper_params(X_train_raw, y_train_raw)
        best_n = int(custom_model_for_cv.n_neighbors)
        best_minkowski = int(custom_model_for_cv.minkowski)
        print(f'Auto-tune attivo: uso k={best_n}, p={best_minkowski} per il training finale.')
    else:
        metrix = custom_validator.validate(X_train_raw, y_train_raw)
        best_n = n
        best_minkowski = minkowski

    sk_metrix = KFoldValidation(
        model=KNeighborsRegressor(n_neighbors=best_n),
        k_folds=k_fold,
        standardize_X=X_standardization,
    ).validate(X_train_raw, y_train_raw)

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
    knn = KNN_Parallel(k=best_n, minkowski=best_minkowski, chunk_size=chunk_size)
    knn_regressor = KNeighborsRegressor(n_neighbors=best_n)

    # Addestramento finale sul training set completo
    start_time = time.time()
    knn.fit(X_train, y_train)
    y_test_pred = knn.predict(X_test)
    end_time = time.time()


    start_time_sk = time.time()
    knn_regressor.fit(X_train, y_train)
    y_test_pred_sk = knn_regressor.predict(X_test)
    end_time_sk = time.time()

    # Valutazione del modello
    print('######## Evaluating my model ########')
    test_metrix = evaluate_model(
        y_true=y_test,
        y_pred=y_test_pred,
        message="Test Set",
        savefile=False,
        path=results_dir,
    )

    print('######## Evaluating sklearn ########')
    test_metrix_sk = evaluate_model(
        y_true=y_test,
        y_pred=y_test_pred_sk,
        message="Test Set",
        savefile=False,
        path=results_dir,
    )

    save_results(test_metrix, 'test_knn_parallel')
    save_results(test_metrix_sk, 'test_knn_sklearn')

    # Visualizzazioni
    # Plot della curva di apprendimento

    plot_predictions(y_test, y_test_pred, model_name="KNN Parallel", assets_dir=assets_dir)
    plot_predictions(y_test, y_test_pred_sk, model_name="KNN sklearn", assets_dir=assets_dir)
    plot_residuals(y_test, y_test_pred, model_name="KNN Parallel", assets_dir=assets_dir)
    plot_residuals(y_test, y_test_pred_sk, model_name="KNN sklearn", assets_dir=assets_dir)
    plot_comparison(metrix, sk_metrix, title='Cross Validation', assets_dir=assets_dir)
    plot_comparison(test_metrix, test_metrix_sk, title='Test', assets_dir=assets_dir)

    # Stampa dei tempi di esecuzione
    print(f'Tempo esecuzione k-NN Parallel: {end_time - start_time}')
    print(f'Tempo esecuzione k-NN sklearn: {end_time_sk - start_time_sk}')