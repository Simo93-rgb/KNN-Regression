import os
from typing import Optional, Tuple 
from ucimlrepo import fetch_ucirepo
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import shuffle


def fetch_data(assets_dir: str = "") -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carica o scarica il dataset e lo restituisce.

    Parameters:
    - assets_dir (str): Directory degli asset.

    Returns:
    - Tuple[pd.DataFrame, pd.Series]: I dati X (features) e y (target).
    """
    if not os.path.exists(f'{assets_dir}/CCPP.csv'):
        combined_cycle_power_plant = fetch_ucirepo(id=17)
        X = combined_cycle_power_plant.data.features
        y = combined_cycle_power_plant.data.targets
        
        # Crea un DataFrame unendo X e y
        df = pd.DataFrame(X, columns=X.columns)
        df['target'] = y
        # Salva in CSV
        csv_file = os.path.join(assets_dir, f'CCPP.csv')
        df.to_csv(csv_file, index=False)
        print(f"Dataset salvato in {csv_file}")
    else:
        df = pd.read_csv(f'{assets_dir}/CCPP.csv')
        X = df.drop(columns='target')
        y = df['target']

        # Converti X e y in valori numerici se ci sono stringhe
        X = X.apply(pd.to_numeric, errors='coerce')
        # X = X.drop(columns=['AP', 'RH'])
        y = pd.to_numeric(y, errors='coerce')

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
    x_scaler = None  # Inizializzo gli scaler a None

    if X_standardization:
        # Standardizzazione delle feature con z-score normalization
        x_scaler = StandardScaler()
        columns = X.columns
        X = x_scaler.fit_transform(X)
        X = pd.DataFrame(X, columns=columns)
        # # Z-Score Normalization
        # X = (X - X.mean(axis=0)) / X.std(axis=0)


    # X, y = remove_outliers_quantile(X, y)
    # Crea un indice casuale per il shuffle
    # indices = np.arange(X.shape[0])
    # np.random.shuffle(indices)
    #
    # # Mescola gli array X e y in base all'indice casuale
    # X = X[indices]
    # y = y[indices]
    # Suddivisione in train (80%) e test (20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if not os.path.exists(f'{assets_dir}/CCPP.csv'):
        # Crea un DataFrame unendo X e y
        df = pd.DataFrame(X, columns=X.columns)
        df['target'] = y
        # Salva in CSV
        csv_file = os.path.join(assets_dir, f'CCPP_standardized.csv')
        df.to_csv(csv_file, index=False)
        print(f"Dataset salvato in {csv_file}")
    


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
