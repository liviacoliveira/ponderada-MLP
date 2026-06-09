import numpy as np
from mlp.activations import relu, softmax

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
        :return: Saída da rede (probabilidades da última camada)
        """
        # Guardamos os valores de Z (antes da ativação) e A (após ativação)
        # para usar depois durante o backpropagation (cálculo de gradientes)
        self.activations = [X]
        self.z_values = []
        
        A_prev = X
        
        # Passa por todas as camadas ocultas
        for i in range(len(self.weights) - 1):
            W = self.weights[i]
            b = self.biases[i]
            
            # Combinação linear: Z = X * W + b
            Z = np.dot(A_prev, W) + b
            self.z_values.append(Z)
            
            # Aplica ReLU nas camadas ocultas
            A = relu(Z)
            self.activations.append(A)
            
            # O output vira o input da próxima camada
            A_prev = A
            
        # Calcula para a última camada (camada de saída)
        W_out = self.weights[-1]
        b_out = self.biases[-1]
        
        Z_out = np.dot(A_prev, W_out) + b_out
        self.z_values.append(Z_out)
        
        # Aplica Softmax na última camada para obter probabilidades
        A_out = softmax(Z_out)
        self.activations.append(A_out)
        
        return A_out
        
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
