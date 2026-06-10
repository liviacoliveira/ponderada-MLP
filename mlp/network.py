import numpy as np
from mlp.activations import relu, softmax, relu_derivative

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
        
    def backward(self, y):
        """
        Backpropagation. Calcula os gradientes baseados no erro (loss).
        
        :param y: Labels verdadeiros em formato one-hot.
        :return: Tupla (dW_list, db_list) com os gradientes dos pesos e biases.
        """
        m = y.shape[0] # tamanho do batch
        
        dW_list = [None] * len(self.weights)
        db_list = [None] * len(self.biases)
        
        # 1. Gradiente da última camada (Softmax + Cross-Entropy)
        # O gradiente da perda em relação ao Z de saída simplifica magicamente para: (Predição - Real)
        A_out = self.activations[-1]
        dZ = A_out - y
        
        # 2. Gradientes dos pesos e biases da última camada
        A_prev = self.activations[-2]
        dW_list[-1] = np.dot(A_prev.T, dZ) / m
        db_list[-1] = np.sum(dZ, axis=0, keepdims=True) / m
        
        # 3. Propagação do erro para trás (camadas ocultas)
        # Caminhamos de trás para frente, ignorando a última que já calculamos
        for i in reversed(range(len(self.weights) - 1)):
            W_next = self.weights[i+1]
            Z_curr = self.z_values[i]
            
            # Erro propagado da camada seguinte
            dA = np.dot(dZ, W_next.T)
            
            # Multiplicamos pela derivada da função de ativação atual (ReLU)
            dZ = dA * relu_derivative(Z_curr)
            
            # Gradientes dos pesos e biases da camada atual
            A_prev_i = self.activations[i]
            dW_list[i] = np.dot(A_prev_i.T, dZ) / m
            db_list[i] = np.sum(dZ, axis=0, keepdims=True) / m
            
        return dW_list, db_list
        
    def train(self, X, y, epochs, batch_size, learning_rate, optimizer='sgd'):
        """
        Executa o loop de treinamento da rede em mini-batches usando SGD ou Adam.

        :param optimizer: Nome do otimizador a usar ('sgd' ou 'adam').
        """
        from mlp.losses import cross_entropy
        from mlp.optimizers import adam, sgd

        m = X.shape[0]
        history = {'loss': [], 'accuracy': []}
        optimizer_state = None
        t = 0

        for epoch in range(epochs):
            # Embaralhar os dados no início de cada época para o mini-batch
            permutation = np.random.permutation(m)
            X_shuffled = X[permutation]
            y_shuffled = y[permutation]
            
            epoch_losses = []
            
            for i in range(0, m, batch_size):
                # Pegar o mini-batch atual
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # 1. Forward Pass
                A_out = self.forward(X_batch)
                
                # 2. Calcular a Loss do batch
                loss = cross_entropy(y_batch, A_out)
                epoch_losses.append(loss)
                
                # 3. Backward Pass (Calcular Gradientes)
                dW_list, db_list = self.backward(y_batch)
                
                # 4. Otimização (Atualizar Pesos in-place com SGD ou Adam)
                optimizer_name = optimizer.lower()
                if optimizer_name == 'adam':
                    optimizer_state, t = adam(
                        self.weights,
                        self.biases,
                        dW_list,
                        db_list,
                        learning_rate,
                        t=t,
                        optimizer_state=optimizer_state,
                    )
                else:
                    sgd(self.weights, self.biases, dW_list, db_list, learning_rate)
                
            # Calcular métricas ao final da época (loss média e acurácia total no treino)
            avg_loss = np.mean(epoch_losses)
            history['loss'].append(avg_loss)
            
            A_train = self.forward(X)
            preds = np.argmax(A_train, axis=1)
            trues = np.argmax(y, axis=1)
            acc = np.mean(preds == trues)
            history['accuracy'].append(acc)
            
            # Print de acompanhamento a cada 10 épocas ou na primeira
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Acc: {acc:.4f}")
                
        return history
