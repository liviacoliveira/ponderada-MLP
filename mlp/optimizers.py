import numpy as np


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


def adam(weights, biases, dW_list, db_list, learning_rate, beta1=0.9, beta2=0.999,
         epsilon=1e-8, t=0, optimizer_state=None):
    """
    Adam: otimizador adicional com momentos de primeira e segunda ordem.

    :param weights: Lista das matrizes de pesos da rede.
    :param biases: Lista dos vetores de bias da rede.
    :param dW_list: Lista dos gradientes dos pesos calculados no backpropagation.
    :param db_list: Lista dos gradientes dos biases calculados no backpropagation.
    :param learning_rate: Taxa de aprendizado.
    :param beta1: Fator de decaimento para o momento de primeira ordem.
    :param beta2: Fator de decaimento para o momento de segunda ordem.
    :param epsilon: Valor pequeno para estabilidade numérica.
    :param t: Passo atual do otimizador.
    :param optimizer_state: Estado interno do Adam para persistir entre mini-batches.
    :return: Tupla (optimizer_state, t) com o estado atualizado e o próximo passo.
    """
    if optimizer_state is None:
        optimizer_state = {
            'm_w': [np.zeros_like(w) for w in weights],
            'v_w': [np.zeros_like(w) for w in weights],
            'm_b': [np.zeros_like(b) for b in biases],
            'v_b': [np.zeros_like(b) for b in biases],
        }

    t += 1

    for i in range(len(weights)):
        m_w = optimizer_state['m_w'][i]
        v_w = optimizer_state['v_w'][i]
        m_b = optimizer_state['m_b'][i]
        v_b = optimizer_state['v_b'][i]

        grad_w = dW_list[i]
        grad_b = db_list[i]

        m_w = beta1 * m_w + (1 - beta1) * grad_w
        v_w = beta2 * v_w + (1 - beta2) * (grad_w ** 2)
        m_b = beta1 * m_b + (1 - beta1) * grad_b
        v_b = beta2 * v_b + (1 - beta2) * (grad_b ** 2)

        optimizer_state['m_w'][i] = m_w
        optimizer_state['v_w'][i] = v_w
        optimizer_state['m_b'][i] = m_b
        optimizer_state['v_b'][i] = v_b

        m_w_hat = m_w / (1 - beta1 ** t)
        v_w_hat = v_w / (1 - beta2 ** t)
        m_b_hat = m_b / (1 - beta1 ** t)
        v_b_hat = v_b / (1 - beta2 ** t)

        weights[i] -= learning_rate * (m_w_hat / (np.sqrt(v_w_hat) + epsilon))
        biases[i] -= learning_rate * (m_b_hat / (np.sqrt(v_b_hat) + epsilon))

    return optimizer_state, t
