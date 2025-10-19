from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
from typing import Dict, Union, Tuple
from knn_parallel import KNN_Parallel
import valutazione


class KFoldValidation:
    def __init__(self, model, k_folds: int = 5) -> None:
        """
        Inizializza la classe di validazione incrociata K-Fold.

        Parameters:
        - model: Il modello da validare.
        - k_folds (int): Il numero di fold per la validazione incrociata.
        """
        self.model: KNN_Parallel = model
        self.k_folds = k_folds
        self.evaluate = valutazione.evaluate_model
        

    def validate_and_find_best_hyper_params(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series],
                                            ) -> Dict[str, float]:
        """
        Esegue la k-fold cross-validation, trova il miglior valore di n_neighbors (k) in base alla RMSE,
        e restituisce le metriche per il k ottimale.

        Parameters:
        - X (np.ndarray | pd.DataFrame): Le features.
        - y (np.ndarray | pd.Series): I target.

        Returns:
        - Dict[str, float]: Un dizionario contenente le metriche medie calcolate per il k ottimale.
        """
        # Converti y in un array numpy se è una Serie Pandas
        y = self._convert_to_numpy(y)
        kf = self._initialize_kfold()
        best_params = self._find_best_params(X, y, kf)

        print(f"Migliori iper parametri: k={best_params['best_k']}, minkowski={best_params['best_minkowski']}")
        print(f"Con RMSE: {best_params['best_mse']}")

        self.model.set_params(n_neighbors=int(best_params['best_k']), minkowski=int(best_params['best_minkowski']))
        return self.validate(X, y)

    def _convert_to_numpy(self, y: Union[np.ndarray, pd.Series]) -> np.ndarray:
        """Converte y in un array numpy se è una Serie Pandas."""
        return np.array(y)

    def _initialize_kfold(self) -> KFold:
        """Inizializza l'oggetto KFold con i parametri specificati."""
        return KFold(n_splits=self.k_folds, shuffle=True, random_state=42)

    def _find_best_params(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, kf: KFold) -> Dict[str, float]:
        """Trova i migliori iperparametri k e minkowski in base alla RMSE usando k-fold cross-validation."""
        best_k = None
        best_minkowski = None
        best_mse = float('inf')
        print(f'Inizio ricerca migliori iper parametri')
        for minkowski in range(1, 4):
            self.model.set_params(minkowski=minkowski)
            for k in range(2, 31):
                self.model.set_params(n_neighbors=k)
                mse = self._cross_validate_model(X, y, kf)
                if mse < best_mse:
                    best_mse = mse
                    best_k = k
                    best_minkowski = minkowski
        print(f'Trovati: neighbourhood={best_k}, minkowski={best_minkowski} con RMSE={best_mse}')
        return {'best_k': best_k, 'best_minkowski': best_minkowski, 'best_mse': best_mse}

    def _cross_validate_model(self, X: Union[np.ndarray, pd.DataFrame], y: np.ndarray, kf: KFold) -> float:
        """Esegue la cross-validation e calcola la RMSE media per un determinato k e minkowski."""
        rmse_array = []
        for train_index, test_index in kf.split(X):
            X_train, X_test = self._select_data(X, train_index, test_index)
            y_train, y_test = y[train_index], y[test_index]
            self.model.fit(X_train, y_train)
            y_pred = self.model.predict(X_test)
            mse = valutazione.root_mean_squared_error(y_test, y_pred)
            rmse_array.append(mse)

        return np.mean(rmse_array)

    def _select_data(self, X: Union[np.ndarray, pd.DataFrame], train_index: np.ndarray, test_index: np.ndarray) -> \
    Tuple[Union[np.ndarray, pd.DataFrame], Union[np.ndarray, pd.DataFrame]]:
        """Seleziona i dati di training e test in base agli indici."""
        if isinstance(X, pd.DataFrame):
            return X.iloc[train_index], X.iloc[test_index]
        else:
            return X[train_index], X[test_index]
    
    def validate(self, X: Union[np.ndarray, pd.DataFrame], y: Union[np.ndarray, pd.Series]) -> Dict[str, float]:
        """
        Esegue la k-fold cross-validation e restituisce la media delle metriche.

        Parameters:
        - X (np.ndarray | pd.DataFrame): Le features.
        - y (np.ndarray | pd.Series): I target.

        Returns:
        - Dict[str, float]: Un dizionario contenente le metriche medie calcolate.
        """
        # Converti y in un array numpy se è una Serie Pandas
        
        y = np.array(y)

        # Inizializzazione delle somme delle metriche
        metric_sums = {}

        # Utilizzo di KFold di scikit-learn
        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        for train_index, test_index in kf.split(X):
            # Utilizzo di .iloc per selezionare le righe con indici posizionali
            if isinstance(X, pd.DataFrame):
                X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            else:
                X_train, X_test = X[train_index], X[test_index]

            y_train, y_test = y[train_index], y[test_index]

            # Addestramento del modello
            self.model.fit(X_train, y_train)

            # Predizione sui dati di test
            y_pred = self.model.predict(X_test)

            # Calcolo delle metriche con evaluate
            metrics = self.evaluate(y_test, y_pred, print_metrix=False, savefile=False)

            # Aggiunta cumulativa delle metriche
            for metric_name, metric_value in metrics.items():
                if metric_name not in metric_sums:
                    metric_sums[metric_name] = 0
                metric_sums[metric_name] += metric_value

        # Calcolo delle metriche medie
        metric_averages = {metric_name: metric_value / self.k_folds for metric_name, metric_value in metric_sums.items()}
        
        return metric_averages


