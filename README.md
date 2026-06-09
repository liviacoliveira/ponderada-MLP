# Multi-Layer Perceptron (MLP) do Zero

Esta é uma implementação de um Multi-Layer Perceptron (MLP) desenvolvida do zero utilizando apenas NumPy, como parte da atividade ponderada da semana 7.

## Como rodar
1. Crie um ambiente virtual e ative-o:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```
2. Instale as dependências:
   ```bash
   pip install -r requirements.txt
   ```
3. Abra o arquivo `notebooks/experimentos.ipynb` no VS Code ou Jupyter, selecione o kernel referente ao `.venv` criado e execute as células.

## Arquitetura escolhida
A arquitetura principal foi construída visando um bom balanço entre poder de representação e tempo de treinamento em CPU:
- **Entrada:** 784 neurônios (vetorização das imagens de 28x28 pixels).
- **Camadas Ocultas:** Duas camadas densas, com 128 e 64 neurônios, respectivamente.
- **Saída:** 10 neurônios (um para cada classe de dígito de 0 a 9).
- **Funções de Ativação:** **ReLU** nas camadas ocultas (rápida de calcular e previne *vanishing gradients*) e **Softmax** na última camada (para saída probabilística).
- **Otimização:** Descida de Gradiente Estocástica (**SGD**) utilizando mini-batches e a função de perda **Cross-Entropy**.

## Resultados
Para atingir os requisitos completos do projeto, realizei comparações de hiperparâmetros/arquitetura:

**Experimento 1 (Arquitetura Principal)**
- **Configuração:** `[784, 128, 64, 10]`, Learning Rate = 0.05, Batch Size = 128, 20 épocas.
- **Acurácia no Teste:** **96,72%** (Acima da meta de 92% estabelecida!).
- **Comportamento:** A curva de Loss caiu rápida e estavelmente, comprovando que a escolha do *learning rate* foi ideal para o *batch size* utilizado.

**Experimento 2 (Comparativo com Rede mais Rasa)**
- **Configuração:** *Apenas uma camada oculta* `[784, 64, 10]`, Learning Rate = 0.05, Batch Size = 128, 20 épocas.
- **Resultado no Teste:** **96,17%**.
- **Conclusão:** Como esperado, a rede com apenas uma camada oculta (e menos neurônios no total) sofreu uma leve queda de desempenho em relação ao modelo com duas camadas ocultas (de 96,72% para 96,17%). Isso demonstra que a profundidade extra do Experimento 1 ajudou a rede a extrair características mais complexas e não lineares das imagens dos dígitos, embora ambos os modelos tenham superado facilmente a meta de 92%.

## Decisões e dificuldades

**Decisão 1: Modularização e começo pelo problema simples**
- Decidi dividir o projeto em módulos separados (`network.py`, `activations.py`, `losses.py`, `optimizers.py`). Acredito que isolar as responsabilidades facilitará o desenvolvimento e, principalmente, a implementação de testes manuais em cada função (como verificar as derivadas separadamente).
- Seguindo a dica do enunciado, pretendo iniciar o teste da classe `MLP` tentando resolver o problema lógico do XOR. Se os gradientes funcionarem para uma rede simples com apenas uma camada oculta no XOR, terei a validação necessária de que a matemática base está correta antes de tentar classificar os 10 dígitos do MNIST.

**Decisão 2: Simplificação matemática na última camada**
- Ao implementar o *backpropagation*, decidi aproveitar o fato de que a derivada da função de perda Cross-Entropy combinada com a ativação Softmax resulta em uma fórmula incrivelmente simples: a diferença entre as probabilidades preditas e os rótulos reais (`Predição - Real`). Implementar dessa forma evitou cálculos desnecessários e garantiu a estabilidade numérica no cálculo do gradiente da última camada.

**Decisão 3: Validação da arquitetura com o teste do XOR**
- Após implementar o loop de treinamento (com *mini-batches* e otimizador *SGD* isolado no arquivo `optimizers.py`), rodei um teste inicial com o problema XOR, conforme havia planejado na Decisão 1. A rede convergiu rapidamente e atingiu 100% de acurácia. Ver a loss caindo confirmou que a matemática da propagação do erro (a regra da cadeia aplicada nas matrizes) estava perfeitamente correta. Isso me deu a confiança necessária para finalmente seguir para o dataset do MNIST.

**Dificuldade 1: Derivadas e o fluxo do Backpropagation**
- Um dos maiores desafios foi garantir que o cálculo analítico dos gradientes na propagação do erro (Backpropagation) estava matematicamente correto. Foi necessário prestar muita atenção e fazer testes detalhados com as dimensões das matrizes durante os produtos escalares (`np.dot`) para que a atualização dos pesos funcionasse sem erros de *shape*. Ajudou muito testar as camadas separadamente antes de integrar na rede completa.

**Dificuldade 2: Escala na Inicialização dos Pesos**
- No início, percebi que se os pesos não fossem inicializados de maneira controlada, a rede poderia ter muita dificuldade em convergir, gerando gradientes que sumiam (*vanishing gradients*) ou explodiam (*exploding gradients*). Ajustar a escala dos números aleatórios gerados pelo NumPy para manter as ativações num limite seguro foi um desafio importante de estabilização do treinamento.