# Multi-Layer Perceptron (MLP) do Zero

Esta é uma implementação de um Multi-Layer Perceptron (MLP) desenvolvida do zero utilizando apenas NumPy, como parte da atividade ponderada da semana 7.

## Como rodar
*Instruções de instalação de dependências e execução do treinamento (a ser preenchido).*

## Arquitetura escolhida
*Detalhes sobre o número de camadas, neurônios, funções de ativação e a justificativa das escolhas (a ser preenchido).*

## Resultados
*Métricas finais de acurácia no MNIST, curvas de loss e comparações de experimentos (a ser preenchido).*

## Decisões e dificuldades

**Decisão 1: Modularização e começo pelo problema simples**
- Decidi dividir o projeto em módulos separados (`network.py`, `activations.py`, `losses.py`, `optimizers.py`). Acredito que isolar as responsabilidades facilitará o desenvolvimento e, principalmente, a implementação de testes manuais em cada função (como verificar as derivadas separadamente).
- Seguindo a dica do enunciado, pretendo iniciar o teste da classe `MLP` tentando resolver o problema lógico do XOR. Se os gradientes funcionarem para uma rede simples com apenas uma camada oculta no XOR, terei a validação necessária de que a matemática base está correta antes de tentar classificar os 10 dígitos do MNIST.
**Decisão 2: Simplificação matemática na última camada**
- Ao implementar o *backpropagation*, decidi aproveitar o fato de que a derivada da função de perda Cross-Entropy combinada com a ativação Softmax resulta em uma fórmula incrivelmente simples: a diferença entre as probabilidades preditas e os rótulos reais (`Predição - Real`). Implementar dessa forma evitou cálculos desnecessários e garantiu a estabilidade numérica no cálculo do gradiente da última camada.

**Decisão 3: Validação da arquitetura com o teste do XOR**
- Após implementar o loop de treinamento (com *mini-batches* e otimizador *SGD* isolado no arquivo `optimizers.py`), rodei um teste inicial com o problema XOR, conforme havia planejado na Decisão 1. A rede convergiu rapidamente e atingiu 100% de acurácia. Ver a loss caindo confirmou que a matemática da propagação do erro (a regra da cadeia aplicada nas matrizes) estava perfeitamente correta. Isso me deu a confiança necessária para finalmente seguir para o dataset do MNIST.

*Mais dificuldades e decisões serão registradas aqui durante a evolução do projeto.*