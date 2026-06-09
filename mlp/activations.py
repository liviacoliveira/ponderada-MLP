import numpy as np

def relu(x):
    """
    Função de ativação ReLU (Rectified Linear Unit).
    Retorna o próprio valor se x > 0, caso contrário retorna 0.
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivada da função ReLU.
    Retorna 1 se x > 0, caso contrário 0.
    """
    return (x > 0).astype(float)

def softmax(x):
    """
    Função Softmax para a camada de saída.
    Converte os logits em probabilidades.
    A subtração de np.max(x) melhora a estabilidade numérica (evita overflow).
    """
    # max() no axis=-1 pega o máximo de cada linha (cada exemplo do batch)
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
