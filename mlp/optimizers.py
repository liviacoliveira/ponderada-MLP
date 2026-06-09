def sgd(weights, biases, dW_list, db_list, learning_rate):
    """
    Stochastic Gradient Descent (SGD) básico.
    Atualiza in-place os pesos e biases da rede com base nos gradientes calculados.
    
    :param weights: Lista das matrizes de pesos da rede.
    :param biases: Lista dos vetores de bias da rede.
    :param dW_list: Lista dos gradientes dos pesos calculados no backpropagation.
    :param db_list: Lista dos gradientes dos biases calculados no backpropagation.
    :param learning_rate: Taxa de aprendizado.
    """
    for i in range(len(weights)):
        weights[i] -= learning_rate * dW_list[i]
        biases[i] -= learning_rate * db_list[i]
