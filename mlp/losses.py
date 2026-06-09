import numpy as np

def cross_entropy(y_true, y_pred):
    """
    Calcula a perda (loss) de Cross-Entropy categórica.
    
    :param y_true: Labels verdadeiros em formato one-hot encoding (batch_size, num_classes)
    :param y_pred: Probabilidades preditas pela rede (batch_size, num_classes)
    :return: Valor numérico da perda média do batch
    """
    # Adicionamos um epsilon pequeno para evitar o log(0) que resulta em NaN (Not a Number)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    
    # Fórmula da entropia cruzada: -sum(y_true * log(y_pred))
    # axis=-1 soma sobre as classes de cada exemplo
    loss = -np.sum(y_true * np.log(y_pred), axis=-1)
    
    # Retorna a média do batch
    return np.mean(loss)
