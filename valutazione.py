from typing import Dict, Union
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, explained_variance_score
import csv
import json
import os



def mean_squared_error(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return np.mean((y_true - y_pred) ** 2)


def root_mean_squared_error(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mean_absolute_error(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return np.mean(np.abs(y_true - y_pred))


def r2_score_metric(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return r2_score(y_true, y_pred)


def explained_variance(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    return explained_variance_score(y_true, y_pred)


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = validate_predictions(y_true, y_pred)
    epsilon = 1e-10  # Per evitare la divisione per zero
    return np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100


def evaluate_model(
        y_true: Union[np.ndarray, pd.Series], 
        y_pred: Union[np.ndarray, pd.Series],
        message: str = "", 
        print_metrix:bool=True, 
        savefile:bool=False,
        path:str=None,
        ) -> Dict[str, float]:
    """
    Calcola varie metriche di errore e accuratezza per il modello, utilizzando i valori predetti e quelli reali.
    Stampa i risultati delle metriche e li restituisce come un dizionario.

    Parameters:
    - y_true (np.ndarray | pd.Series): I valori target effettivi.
    - y_pred (np.ndarray | pd.Series): I valori target predetti dal modello.
    - message (str): Messaggio opzionale da aggiungere alla descrizione stampata delle metriche.
    - print_metrix (bool): Se True stampa i valori, altrimenti no.

    Returns:
    - Dict[str, float]: Un dizionario contenente le metriche calcolate, con le seguenti chiavi:
        - "MSE": Mean Squared Error.
        - "RMSE": Root Mean Squared Error.
        - "MAE": Mean Absolute Error.
        - "R2": Coefficiente di Determinazione (R² Score).
        - "ExVar": Explained Variance Score.
        - "MAPE": Mean Absolute Percentage Error.
    """
    if path is None:
        # Percorso del file main.py
        current_dir = os.path.dirname(os.path.abspath(__file__))

        # Percorso alla cartella "Assets" nella directory "KNN Regression"
        assets_dir = os.path.join(current_dir, 'Assets')
        path=os.path.join(assets_dir, 'results')

    # Convalida le predizioni e i valori target
    y_true, y_pred = validate_predictions(y_true, y_pred)
    
    # Calcola le metriche
    metrix = {"RMSE": root_mean_squared_error(y_true, y_pred),
              "MAE": mean_absolute_error(y_true, y_pred), "R2": r2_score_metric(y_true, y_pred),
              "ExVar": explained_variance(y_true, y_pred), "MAPE": mean_absolute_percentage_error(y_true, y_pred)}

    if print_metrix:
        # Configura il messaggio aggiuntivo
        if message:
            message = "su " + message
        # Stampa le metriche calcolate
        print(f"Root Mean Squared Error (RMSE) {message}:", metrix['RMSE'])
        print(f"Mean Absolute Error (MAE) {message}:", metrix['MAE'])
        print(f"R² Score {message}:", metrix['R2'])
        print(f"Explained Variance Score {message}:", metrix['ExVar'])
        print(f"Mean Absolute Percentage Error (MAPE) {message}:", metrix['MAPE'])
    if savefile:
        if os.path.exists(path):
            with open(f'{path}/{message}.csv', 'w', newline='') as file:
                writer = csv.writer(file)
                writer.writerow(metrix.keys())  # Scrive le intestazioni
                writer.writerow(metrix.values())  # Scrive i valori
        
            # Salvataggio in un file JSON
            with open(f'{path}/{message}.json', 'w') as file:
                json.dump(metrix, file, indent=4)

    return metrix


def validate_predictions(y_true, y_pred):
    # Verifica che y_true e y_pred siano array NumPy, liste o Serie/DataFrame Pandas
    assert isinstance(y_true, (list, np.ndarray, pd.Series,
                               pd.DataFrame)), f"Errore: y_true deve essere una lista, un array NumPy, una Serie o un DataFrame Pandas, ma è {type(y_true)}"
    assert isinstance(y_pred, (list, np.ndarray, pd.Series,
                               pd.DataFrame)), f"Errore: y_pred deve essere una lista, un array NumPy, una Serie o un DataFrame Pandas, ma è {type(y_pred)}"

    # Se y_true o y_pred sono DataFrame, estrai la prima colonna
    if isinstance(y_true, pd.DataFrame):
        y_true = y_true.iloc[:, 0].values
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.iloc[:, 0].values

    # Se y_true o y_pred sono Serie Pandas, converti in NumPy array
    if isinstance(y_true, pd.Series):
        y_true = y_true.values
    if isinstance(y_pred, pd.Series):
        y_pred = y_pred.values

    # Converti in array NumPy se sono liste
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    # Verifica che abbiano la stessa lunghezza
    assert len(y_true) == len(
        y_pred), f"Errore: y_true e y_pred devono avere la stessa lunghezza. Lunghezza di y_true: {len(y_true)}, Lunghezza di y_pred: {len(y_pred)}"

    # Verifica che i tipi di dato siano consistenti
    assert y_true.dtype == y_pred.dtype, f"Errore: y_true e y_pred devono essere dello stesso tipo. Tipo di y_true: {y_true.dtype}, Tipo di y_pred: {y_pred.dtype}"

    # Verifica che y_true e y_pred contengano solo valori numerici
    assert np.issubdtype(y_true.dtype, np.number), "Errore: y_true deve contenere solo valori numerici."
    assert np.issubdtype(y_pred.dtype, np.number), "Errore: y_pred deve contenere solo valori numerici."

    # Verifica che non ci siano NaN nei dati
    assert not np.isnan(y_true).any(), "Errore: y_true contiene valori NaN."
    assert not np.isnan(y_pred).any(), "Errore: y_pred contiene valori NaN."

    # Verifica che non ci siano valori infiniti
    assert np.isfinite(y_true).all(), "Errore: y_true contiene valori infiniti."
    assert np.isfinite(y_pred).all(), "Errore: y_pred contiene valori infiniti."

    # Se tutti gli assert passano, ritorna gli array validati
    # print("Tutti i controlli sono stati superati. y_true e y_pred sono validi.")
    return y_true, y_pred
