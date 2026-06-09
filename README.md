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

*Mais dificuldades e decisões serão registradas aqui durante a evolução do projeto.*