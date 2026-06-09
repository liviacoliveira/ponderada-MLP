import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        """
        Inicializa a rede neural (Multi-Layer Perceptron).
        
        :param layer_sizes: Lista com o número de neurônios em cada camada.
                            Ex: [2, 4, 1] para o XOR (2 entradas, 4 ocultos, 1 saída).
                            Ex: [784, 128, 64, 10] para o MNIST.
        """
        self.layer_sizes = layer_sizes
        self.weights = []
        self.biases = []
        
        # Inicialização dos parâmetros (pesos e biases)
        # i vai de 0 até (número de camadas - 2)
        for i in range(len(layer_sizes) - 1):
            # Usando uma inicialização aleatória com média 0 e um pequeno desvio padrão
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * 0.01
            b = np.zeros((1, layer_sizes[i+1]))
            
            self.weights.append(w)
            self.biases.append(b)
            
    def forward(self, X):
        """
        Passo forward da rede. Calcula a saída passando os dados 
        por todas as camadas da rede.
        
        :param X: Dados de entrada, formato (batch_size, num_features).
        """
        # TODO: Implementar a passagem para frente
        # (X * W + b) -> Ativação
        pass
        
    def backward(self, X, y, learning_rate):
        """
        Backpropagation. Calcula os gradientes baseados no erro (loss)
        e atualiza os pesos da rede.
        
        :param X: Dados de entrada.
        :param y: Labels verdadeiros / Esperados.
        :param learning_rate: Taxa de aprendizado.
        """
        # TODO: Implementar a passagem para trás (regra da cadeia)
        pass
        
    def train(self, X, y, epochs, batch_size, learning_rate):
        """
        Executa o loop de treinamento da rede em mini-batches usando SGD.
        """
        # TODO: Implementar loop de treinamento
        pass
