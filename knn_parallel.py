from typing import Optional, Union
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

class KNN_Parallel:
    def __init__(self, k: int = 3, n_jobs: int = -1, minkowski:int = 2, chunk_size:int=1) -> None:
        """
        Inizializza il modello KNN.

        Parameters:
        - n_neighbors(int): Numero di vicini da considerare.
        - n_jobs (int): Numero di lavori paralleli da eseguire. -1 per usare tutti i core.
        """
        self.n_neighbors = k
        self.minkowski = minkowski
        self.n_jobs = n_jobs
        self.chunk_size = chunk_size
        self.X_train: Optional[np.ndarray] = None
        self.y_train: Optional[np.ndarray] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Addestra il modello KNN.

        Parameters:
        - X_train (np.ndarray): Il set di dati di training.
        - y_train (np.ndarray): I target di training.
        """
        self.X_train = X_train
        self.y_train = y_train

    def _euclidean_distance(self, x1: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Calcola la distanza euclidea tra x1 e tutte le righe di X_train.

        Parameters:
        - x1 (np.ndarray): Un singolo punto dati.
        - X_train (np.ndarray): Il set di dati di training.

        Returns:
        - np.ndarray: Un array di distanze.
        """
        return np.sqrt(np.sum((X_train - x1) ** 2, axis=1))
    
    def _minkowski(self, x1: np.ndarray, X_train: np.ndarray) -> np.ndarray:
        """
        Calcola la distanza di Minkowski tra un punto x1 e tutti i punti di X_train.

        Parameters:
        - x1 (np.ndarray): Un singolo punto dati.
        - X_train (np.ndarray): Il set di dati di training.

        Returns:
        - np.ndarray: Un array di distanze di Minkowski.
        """
        return np.sum(np.abs(X_train - x1) ** self.minkowski, axis=1) ** (1 / self.minkowski)


    def _predict_single(self, x: np.ndarray) -> float:
        """
        Predice il valore per un singolo punto dati.

        Parameters:
        - x (np.ndarray): Un singolo punto dati.

        Returns:
        - float: La predizione per il punto dati.
        """
        X_train_array = np.array(self.X_train)
        y_train_array = np.array(self.y_train)

        # Calcolo delle distanze dal punto x a tutti i punti di training
        # distances = self._euclidean_distance(x, X_train_array)
        distances = self._minkowski(x, X_train_array)

        # Identifica gli indici dei n_neighborsvicini più vicini
        k_indices = np.argsort(distances)[:self.n_neighbors]

        # Raccoglie i target dei n_neighborsvicini (da y_train_array, che è ora un array numpy)
        k_nearest_targets = y_train_array[k_indices]

        # Calcolo dei pesi usando l'inverso del quadrato della distanza (aggiungo epsilon per evitare divisioni per zero)
        epsilon = 1e-10
        k_nearest_distances = distances[k_indices]
        weights = np.where(k_nearest_distances < 1, 1 - k_nearest_distances, 1 / (k_nearest_distances ** 2 + epsilon))

        # Normalizzazione dei pesi
        weights_normalized = weights / np.sum(weights)

        # Calcolo della predizione come media pesata
        return np.sum(k_nearest_targets * weights_normalized)

    def divide_chunks(self, data: np.ndarray, n: int) -> list:
        """
        Divide l'array `data` in blocchi di dimensione `n`.

        Parameters:
        - data (np.ndarray): L'array da dividere.
        - n (int): La dimensione di ciascun blocco.

        Returns:
        - list: Una lista di blocchi (ogni blocco è un sotto-array).
        """
        for i in range(0, len(data), n):
            yield data[i:i + n]

    def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Predice i valori per i nuovi dati basandosi sul KNN con pesatura delle distanze.
        Esegue due predizioni per job per migliorare l'efficienza.

        Parameters:
        - X_test (np.ndarray | pd.DataFrame): I dati di test.
        - chunk_size (int): Definisce la dimensione del blocco di predizioni per ogni job.

        Returns:
        - np.ndarray: Le predizioni.
        """
        # Se X_test è un DataFrame, convertilo in un array numpy
        if isinstance(X_test, pd.DataFrame):
            X_test = X_test.values

        # Divide `X_test` in blocchi di dimensione `chunk_size`
        blocks = list(self.divide_chunks(X_test, self.chunk_size))

        # Parallelizza la predizione su ciascun blocco
        predictions = Parallel(n_jobs=self.n_jobs)(
            delayed(lambda block: [self._predict_single(x) for x in block])(block) for block in blocks
        )

        # Converte i risultati in un array numpy appiattendo i risultati di ciascun blocco
        return np.array([pred for block_preds in predictions for pred in block_preds])

    # def predict(self, X_test: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
    #     """
    #     Predice i valori per i nuovi dati basandosi sul KNN con pesatura delle distanze.
    #
    #     Parameters:
    #     - X_test (np.ndarray | pd.DataFrame): I dati di test.
    #
    #     Returns:
    #     - np.ndarray: Le predizioni.
    #     """
    #     # Se X_test è un DataFrame, convertilo in un array numpy
    #     if isinstance(X_test, pd.DataFrame):
    #         X_test = X_test.values
    #
    #     # Utilizza joblib per parallelizzare la predizione
    #     predictions = Parallel(n_jobs=self.n_jobs)(
    #         delayed(self._predict_single)(x) for x in X_test
    #     )
    #
    #     return np.array(predictions)

    def set_params(self, n_neighbors: Optional[int] = None, n_jobs: Optional[int] = None, minkowski:Optional[int] = None) -> None:
        """
        Imposta i parametri del modello KNN.

        Parameters:
        - n_neighbors(Optional[int]): Il nuovo valore di n_neighbors(numero di vicini). Se None, mantiene il valore attuale.
        - n_jobs (Optional[int]): Il nuovo valore di n_jobs (numero di lavori paralleli). Se None, mantiene il valore attuale.
        """
        if n_neighbors is not None:
            self.n_neighbors = n_neighbors
        if n_jobs is not None:
            self.n_jobs = n_jobs
        if minkowski is not None:
            self.minkowski = minkowski

