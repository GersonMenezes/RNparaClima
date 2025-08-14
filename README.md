# Rede Neural para Previsão Climática

Este projeto implementa uma rede neural feedforward em C puro para previsão de condições climáticas, especificamente sensação térmica e probabilidade de chuva

## 📋 Requisitos

- **Compilador**: GCC com suporte a C99
- **Bibliotecas**: math.h (incluir `-lm` na compilação)
- **Sistema**: Linux/Unix, Windows (com MinGW), macOS
- 
  ### Preparação do ambiente e instalações necessárias
## É necessário ter o python instalado e adicionado ao path do sistema
## Também é necessário ter a biblioteca matplotlib instalada, ela é facilmente instalado com o comando ##abaixo
```bash
pip install matplotlib
```

## 📁 Estrutura do Projeto

```
projeto/
├── run_and_plot.py    # Treina modelo e exibe gráfico
├── main.c             # Código principal
├── train.csv           # Dados de treinamento
└── README.md           # Este arquivo
```

##  Compilação e Execução


### Compilação e Treinamento do modelo
##Enquanto o modelo é treinado será possível ver o gráfico de treinamento:
```bash
python run_and_plot.py
```
```

### 🚀 Executar o programa com o modelo treinado e colocar valores de entrada:
```bash
./central_comando --load-model
```
```
### Variáveis de Entrada
1. **Temperatura** (°C)
2. **Umidade** (%)
3. **Vento** (km/h)
4. **Hora** (0-23)
5. **Dia** (1-365)
6. **Nuvens** (%)
7. **Pressão** (hPa)
8. **Precipitação** (mm)

### Variáveis de Saída
1. **Sensação térmica** (°C)
2. **Probabilidade de chuva** (%)


**Importante**: Certifique-se de que o arquivo `train.csv` está no mesmo diretório do executável.

## 📊 Formato dos Dados

O arquivo `train.csv` deve conter os dados no seguinte formato:

```csv
25.5,60,15,14,120,30,1013,0,24.2,15.5
22.1,80,8,9,85,70,1008,2.1,20.8,75.2
...
```

## 📈 Processo de Treinamento

1. **Carregamento**: Lê dados do arquivo CSV
2. **Normalização**: Calcula estatísticas (média/desvio) e normaliza dados usando Z-score
3. **Treinamento**: 
   - Embaralha dados a cada época (se `SHUFFLE = 1`)
   - Executa feedforward e backpropagation
   - Monitora a perda (MSE) por época
4. **Interação**: Permite previsões interativas após o treinamento

## 💡 Exemplo de Uso

Após o treinamento, o programa solicitará entrada interativa:

```
Temperatura: 25.5
Umidade: 65
Vento: 12
Hora: 14
Dia: 180
Nuvens: 40
Pressao: 1013
Precipitacao: 0
```

A saída vai se parecer como:
```
Sensacao: 24.80C, Probabilidade de Chuva: 25.30%
```

## 🔧 Funcionalidades Técnicas

### Normalização Z-Score
```c
zscore(x) = (x - média) / desvio_padrão
```

## ⚙️ Configurações

| Parâmetro | Valor | Descrição |
|-----------|-------|-----------|
| `INPUT_NODES` | 8 | Neurônios na camada de entrada |
| `HIDDEN_NODES` | 16 | Neurônios na camada oculta |
| `OUTPUT_NODES` | 2 | Neurônios na camada de saída |
| `EPOCHS` | 600 | Número de épocas de treinamento |
| `LEARNING_RATE` | 0.001 | Taxa de aprendizado |
| `MAX_DATA` | 100,000 | Máximo de amostras suportadas |

### Função ReLU
```c
relu(x) = max(0, x)
```

### Algoritmo de Treinamento
- **Forward pass**: Propaga inputs através da rede
- **Backward pass**: Calcula gradientes e atualiza pesos
- **Embaralhamento**: Melhora a generalização

- ## 📋 Características

- **Arquitetura**: Rede neural feedforward de 3 camadas
- **Entrada**: 8 variáveis climáticas
- **Saída**: 2 previsões (sensação térmica e probabilidade de chuva)
- **Função de ativação**: ReLU na camada oculta
- **Normalização**: Z-score para inputs e outputs
- **Treinamento**: Backpropagation com embaralhamento de dados

## 🏗️ Arquitetura da Rede

```
Camada de Entrada (8 neurônios)
    ↓
Camada Oculta (16 neurônios + ReLU)
    ↓
Camada de Saída (2 neurônios)
```

## 🎯 Possíveis Melhorias

- [ ] Implementar validação cruzada
- [ ] Adicionar regularização (L1/L2)
- [ ] Salvar/carregar modelo treinado
- [ ] Interface gráfica para visualização
- [ ] Suporte a diferentes funções de ativação
- [ ] Implementar early stopping
- [ ] Adicionar métricas de avaliação (R², MAE)

## 📝 Notas de Implementação

- Os pesos são inicializados aleatoriamente entre -1 e 1
- A rede usa gradiente descendente simples (sem momentum)
- Dados são embaralhados a cada época para melhor convergência
- A normalização é essencial para o bom funcionamento da rede

## 🐛 Troubleshooting

**Perda não converge**
- Ajuste a taxa de aprendizado (`LEARNING_RATE`)
- Verifique a qualidade dos dados de entrada
- Considere aumentar o número de épocas

**Previsões inconsistentes**
- Verifique se os dados de entrada estão na mesma escala do treinamento
- Confirme que todas as 8 variáveis estão sendo fornecidas corretamente

## 📄 Licença

Este projeto está sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.