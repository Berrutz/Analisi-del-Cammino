import numpy as np


def shift(X: list | np.ndarray, n: int) -> list | np.ndarray:
    """ Shifta di n periodi la funzione X
    """

    # controllo per n
    if not isinstance(n, int):
        raise Exception('n deve essere un valore intero')

    X_copy = X.copy() # copia e rendi numpy array
    if not isinstance(X_copy, np.ndarray):
        X_copy = np.array(X_copy)

    # se si shifta troppo impostiamo n
    # alla lunghezza della funzione X
    if np.abs(n) > len(X):
        n = np.sign(n) * len(X)
    
    # dato  n shiftiamo la funzione verso sinistra
    # dato -n shiftiamo la funzione verso destra
    for i in range(np.abs(n)):
        # togliamo il primo o lultimo valore
        # in base a dove vogliamo shiftare
        X_copy = X_copy[1:] if np.sign(n) == 1 else X_copy[:-1]

        # shifta la funzione aggiungendo uno 
        # 0 in testa o in coda in base a dove
        # vogliamo shiftare
        X_copy = np.insert(X_copy, 0 if np.sign(n) == -1 
            else len(X_copy), 0)

    return X_copy.tolist() if not isinstance(X, np.ndarray) \
        else np.array(X_copy)


def singola_autocorrelazione(X: list | np.ndarray, tau: int, normalizzazione = True) -> float:

    """ Esegue una signola autocorrelazione data la latenza
        tau
    """

    # controllo per tau
    if not isinstance(tau, int) and tau < 0:
        raise Exception('tau deve essere un intero \
            maggiore di 0')

    # controllo che X sia un array numpy
    if not isinstance(X, np.ndarray):
        X = np.array(X)


    if normalizzazione:
        mean = X.mean()
        var = X.var()
        
        # f(t-tau) - mu
        primo_termine   = shift(X - mean, -tau)

        # coniugato(f(t)- mu)
        secondo_termine = np.conjugate(X - mean)

        #                                           ... / sigma^2
        return np.mean( primo_termine * secondo_termine ) / var

    # espettazione[ f(t-tau)       * coniugato(f(t)) ]   
    return np.mean( shift(X, -tau) * np.conjugate(X) )


def funzione_autocorrelazione(X: list | np.ndarray, normalizzazione = True) -> list | np.ndarray:
    
    """ calcola la funzione di autocorrelazione
    """
    autocorrelazione: list = []
    # calcola la autocorrelazione singola per ogni
    # possibile tau
    for i in range(len(X)):
        autocorrelazione.append(singola_autocorrelazione(X, i, normalizzazione=normalizzazione))

    return autocorrelazione if not isinstance(X, np.ndarray) \
        else np.array(autocorrelazione)

