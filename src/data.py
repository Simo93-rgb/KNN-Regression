import os
from typing import Optional, Tuple
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np


def fetch_data(assets_dir: str = "") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carica o scarica il dataset e lo restituisce.

    Parameters:
    - assets_dir (str): Directory degli asset.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: I dati X (features) e y (target).
    """
    csv_file = os.path.join(assets_dir, 'CCPP.csv')

    if os.path.exists(csv_file):
        df = pd.read_csv(f'{assets_dir}/CCPP.csv')
        X = df.drop(columns='target')
        y = df['target']

        # Converti X e y in valori numerici se ci sono stringhe
        X = X.apply(pd.to_numeric, errors='coerce')
        y = pd.to_numeric(y, errors='coerce')

        valid_rows = (~X.isna().any(axis=1)) & (~y.isna())
        X = X.loc[valid_rows]
        y = y.loc[valid_rows]

        if len(X) > 0:
            return X, y

        print('CCPP.csv non valido per regressione: rigenero il dataset CCPP corretto.')

    # UCI CCPP dataset id for ucimlrepo.
    combined_cycle_power_plant = fetch_ucirepo(id=294)
    X = combined_cycle_power_plant.data.features
    y = combined_cycle_power_plant.data.targets
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    valid_rows = (~X.isna().any(axis=1)) & (~y.isna())
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    df = pd.DataFrame(X, columns=X.columns)
    df['target'] = y
    df.to_csv(csv_file, index=False)
    print(f"Dataset salvato in {csv_file}")

    return X, y


def edit_dataset(
        X: pd.DataFrame, 
        y: pd.Series, 
        X_standardization: bool = True, 
        test_size=0.2,
        assets_dir=""
        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler]]:
    """
    Standardizza il dataset e lo suddivide in training e test.

    Parameters:
    - X (pd.DataFrame): Le features.
    - y (pd.Series): Il target.
    - X_standardization (bool): Se standardizzare o meno X.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Optional[StandardScaler], Optional[StandardScaler]]:
    X_train, X_test, y_train, y_test, e gli scaler (se usati).
    """
    x_scaler = None

    # Difesa aggiuntiva: garantisce input numerico e senza NaN anche quando
    # edit_dataset viene chiamata con dati non passati da fetch_data.
    X = X.apply(pd.to_numeric, errors='coerce')
    y = pd.to_numeric(y, errors='coerce')
    valid_rows = (~X.isna().any(axis=1)) & (~y.isna())
    dropped_rows = int((~valid_rows).sum())
    if dropped_rows > 0:
        print(f"Rimosse {dropped_rows} righe con valori non numerici/NaN prima dello split.")
    X = X.loc[valid_rows]
    y = y.loc[valid_rows]

    if len(X) == 0:
        raise ValueError("Il dataset e vuoto dopo la rimozione di valori non numerici/NaN.")

    # Split prima del preprocessing: evita data leakage.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
    )

    if X_standardization:
        x_scaler = StandardScaler()
        columns = X_train.columns
        X_train = pd.DataFrame(x_scaler.fit_transform(X_train), columns=columns, index=X_train.index)
        X_test = pd.DataFrame(x_scaler.transform(X_test), columns=columns, index=X_test.index)

        if assets_dir:
            standardized_path = os.path.join(assets_dir, 'CCPP_standardized.csv')
            if not os.path.exists(standardized_path):
                X_full = pd.DataFrame(x_scaler.transform(X), columns=columns, index=X.index)
                df = X_full.copy()
                df['target'] = y
                df.to_csv(standardized_path, index=False)
                print(f"Dataset salvato in {standardized_path}")
    


    return X_train, X_test, y_train, y_test, x_scaler


def remove_outliers_quantile(X: pd.DataFrame, y: pd.Series, lower_quantile: float = 0.01, upper_quantile: float = 0.99) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Rimuove outliers dai dati in base ai quantili.
    
    Parameters:
    - X (pd.DataFrame): Le features.
    - y (pd.Series): Il target.
    - lower_quantile (float): Il quantile inferiore per tagliare outliers.
    - upper_quantile (float): Il quantile superiore per tagliare outliers.
    
    Returns:
    - Tuple[pd.DataFrame, pd.Series]: I dati senza outliers.
    """
    quantiles = X.quantile([lower_quantile, upper_quantile])
    
    # Filtro le righe che sono all'interno dei quantili
    filtered_entries = (X >= quantiles.loc[lower_quantile]) & (X <= quantiles.loc[upper_quantile])
    filtered_entries = filtered_entries.all(axis=1)  # Ottieni righe dove tutte le colonne rispettano il filtro
    
    return X[filtered_entries], y[filtered_entries]
