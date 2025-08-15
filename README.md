# Projeto CLP — Integração: C (cálculo) + Python (interface)

Nome:Eduardo Timm Buss, Gerson Menezes

**Descrição curta:**  
Exemplo didático que integra duas linguagens: **C** (implementa e treina uma pequena rede neural) e **Python** (orquestra a execução, lê saída em tempo real e plota a curva de loss). O objetivo é demonstrar a *linkagem* entre linguagens distintas, com responsabilidades separadas.

---

## Conteúdo deste README (navegação rápida)
1. [O que tem no repositório](#o-que-tem-no-repositório)  
2. [Pré-requisitos / dependências](#pré-requisitos--dependências)  
3. [Arquivos principais e o que cada um faz](#arquivos-principais-e-o-que-cada-um-faz)  
4. [Formato dos dados de entrada (`train.csv`)](#formato-dos-dados-de-entrada-traincsv)  
5. [Arquitetura da rede (detalhada)](#arquitetura-da-rede-detalhada)  
6. [Como compilar / executar (passo a passo)](#como-compilar--executar-passo-a-passo)  
7. [Makefile (conteúdo sugerido)](#makefile-conteúdo-sugerido)  
8. [Como a integração (linkagem) entre C e Python funciona](#como-a-integração-linkagem-entre-c-e-python-funciona)  
9. [Formato da saída e do `model.bin` (explicação)](#formato-da-saída-e-do-modelbin-explicação)  
10. [Exemplos de execução e saída esperada](#exemplos-de-execução-e-saída-esperada)  
11. [Soluções para problemas comuns / troubleshooting](#soluções-para-problemas-comuns--troubleshooting)  

---

## O que tem no repositório
- `main.c` — implementação em C da rede neural + rotina de treino e persistência (gera `model.bin` e emite progresso no `stdout`).
- `run_and_plot.py` — script Python que (se necessário) compila `main.c`, executa o binário, lê sua saída em tempo real e plota a curva de loss por época.
- `Make.mk` — com os targets mínimos:
  - `make run` — inicia `python run_and_plot.py`
  - `make clear` — remove executáveis gerados (`central_comando` / `central_comando.exe`)
- `train.csv` — arquivo de dados de treino que você deve fornecer (formato descrito abaixo).
- `model.bin` — arquivo gerado pelo programa em C ao salvar o modelo (após o treino).
- `README.md` — este arquivo.

---

## Pré-requisitos / dependências

### Software principal
- Python 3.x (recomendado 3.8+)
- (Opcional para compilar o C manualmente) GCC (Linux/WSL/Git Bash) ou MinGW/MSYS2 (Windows)

### Bibliotecas Python (necessárias para o `run_and_plot.py`)
- `matplotlib` — **necessário** para plot em tempo real.
- `numpy` — utilidades numéricas (opcional, mas recomendado).

Instalação recomendada:
```bash
pip install matplotlib numpy
```

> **Observação importante:** se você não instalar `matplotlib`, o `run_and_plot.py` não conseguirá mostrar o gráfico em tempo real. Se você só quer executar o binário C sem Python, `matplotlib` não é necessário.

---

## Arquivos principais e o que cada um faz (detalhado)

### `main.c`
- Implementa uma rede neural feedforward de topologia fixa (ver seção “Arquitetura da rede”).
- Lê dados de treino do arquivo `train.csv` (formato explicado abaixo).
- Normaliza dados por **z-score** (média e desvio padrão).
- Treina a rede por um número fixo de épocas (definido em `main.c`).
- Durante o treino imprime linhas no `stdout` com a loss por época para monitoramento em tempo real (`LOSS:<epoca>,<valor>`).
- Ao final do treino salva os pesos/bias em `model.bin` e imprime `TRAINING_DONE`.

### `run_and_plot.py`
- Tarefa: controlar/monitorar/exibir.
- Funcionalidades:
  - Se o binário (`central_comando`) não existir, tenta compilar `main.c`.
  - Executa o binário como um processo filho (`subprocess.Popen`) e lê o `stdout` linha a linha.
  - Faz parse das linhas `LOSS:` para construir uma lista de losses e plota com `matplotlib` em tempo real.
  - Ao receber `TRAINING_DONE`, encerra plot e exibe o resultado final.
- Observação: este script é a “interface” do trabalho — mostra como Python pode orquestrar e visualizar o trabalho que o C faz.

### `Make.mk`
- Contém targets mínimos `run` e `clear` para facilitar execução e limpeza (ver seção Makefile abaixo).

---

## Formato dos dados de entrada (`train.csv`)

O `main.c` espera um CSV sem cabeçalho, cada linha representando 1 amostra.

- Cada linha deve ter **10 valores** separados por vírgula (`,`):
  - **8 primeiros valores** = features de entrada (x₁..x₈)
  - **2 últimos valores** = targets/labels (y₁,y₂)

**Exemplo de `train.csv` (3 amostras fictícias):**
```
0.12,-1.3,2.0,0.0,5.1,-0.2,3.3,0.01,1.0,0.0
-0.5,0.0,1.2,0.3,-0.1,2.2,0.4,-0.9,0.0,1.0
1.0,0.5,-0.2,0.2,0.1,0.0,-1.1,0.9,1.0,0.0
```

**Regras:**
- Não inclua cabeçalho (nomes das colunas).
- Separe valores com `,`.
- Cada linha **obrigatoriamente** deve ter 10 valores; linhas mal formatadas podem ser ignoradas ou causar erro dependendo da validação em `main.c`.

---

## Arquitetura da rede (detalhada)
A rede implementada no `main.c` segue os parâmetros abaixo (conforme o código que você enviou):

- **Topologia:** `8 (entrada) → 16 (camada oculta) → 2 (saída)`
- **Ativações:** (depende da implementação no `main.c`; normalmente a camada oculta usa tanh/sigmoid/ReLU e a saída linear/softmax conforme problema — consulte comentários no `main.c` para exatamente qual foi usado).
- **Pré-processamento:** normalização por **z-score** (para cada feature: subtrai média e divide pelo desvio padrão calculados no dataset).
- **Função de perda:** Mean Squared Error (MSE) média por amostra — o valor impresso por época.
- **Treino:** algoritmo de atualização direta (gradiente ou versão simples implementada no `main.c`). Parâmetros como taxa de aprendizagem, número de épocas (`EPOCHS`), momentum, etc., são constantes em `main.c` e podem ser alterados editando o arquivo e recompilando.
- **Persistência:** modelo salvo em `model.bin` ao final do treinamento.

> **Dica:** se quiser experimentar outras topologias ou hiperparâmetros, altere `main.c` e recompile. Mantenha cópias do `model.bin` se quiser comparar resultados.

---

## Como compilar / executar — passo a passo

### 1) Preparar ambiente Python (recomendado)
Crie um virtualenv (opcional, mas recomendado):
```bash
python -m venv .venv
# ativar
# Linux / macOS:
source .venv/bin/activate
# Windows PowerShell:
.venv\\Scripts\\Activate.ps1
# Windows CMD:
.venv\\Scripts\\activate.bat
```

Instale dependências:
```bash
pip install matplotlib numpy
```

### 2) Usando Make (Linux / Git Bash / WSL / MSYS2)
**Executar (inicia o script Python):**
```bash
make run
```

**Limpar executáveis:**
```bash
make clear
```

> Obs: o `make` exige que você tenha o utilitário `make`. No Linux ele geralmente já existe. No Windows, use Git Bash, MSYS2, ou WSL para ter `make`.

### 3) Sem Make — comandos diretos

**Compilar manualmente (opcional):**
```bash
gcc -O3 -o central_comando main.c -lm
```
(Se você usar MinGW/MSYS2 no Windows, o `gcc` funciona de forma similar.)

**Executar apenas o Python (o `run_and_plot.py` compila automaticamente se desejar):**
```bash
python run_and_plot.py
```

**Executar o binário diretamente:**
```bash
# Linux / macOS
./central_comando

# Windows (CMD / PowerShell)
central_comando.exe
```


## Como a integração (linkagem) entre C e Python funciona 

Vou descrever com passos numerados o fluxo exato que ocorre quando você executa `make run` (ou `python run_and_plot.py`).

1. **Iniciar o controlador (Python):**
   - Você roda `python run_and_plot.py` (ou `make run` que chama o mesmo).
   - O Python verifica se o binário do C (`central_comando`) existe; se não existir, tenta compilar `main.c` usando `gcc`.

2. **Compilação (se necessária):**
   - `python` chama algo como `gcc -O3 -o central_comando main.c -lm`.
   - Se a compilação falhar, o Python exibe o `stderr` e aborta; caso contrário, prossegue.

3. **Execução do binário C como processo filho:**
   - Python executa `./central_comando` (ou `central_comando.exe` no Windows) via `subprocess.Popen`, com `stdout` e `stderr` redirecionados para pipes de onde o Python pode ler.

4. **Treinamento em C e mensagens de progresso:**
   - Enquanto o C treina a rede, ele imprime periodicamente no `stdout` mensagens no formato:
     ```
     LOSS:<epoca>,<valor>
     ```
     Ex.: `LOSS:10,0.012345678`
   - Ao terminar o treino, o C imprime:
     ```
     TRAINING_DONE
     ```
     (e, possivelmente, outras mensagens de confirmação).

5. **Python lê as mensagens em tempo real:**
   - Em uma thread separada, `run_and_plot.py` lê o `stdout` linha a linha.
   - Para cada linha que começa com `LOSS:`, o Python:
     - faz `split` / parse da linha para extrair época e valor numérico,
     - adiciona o valor a uma lista interna (`losses`),
     - atualiza o gráfico (`matplotlib`) com os novos dados (plot em tempo real).
   - Se receber `TRAINING_DONE`, o Python:
     - atualiza a interface (indica fim),
     - fecha/junta o processo (wait), e
     - salva/mostra o plot final.

6. **Persistência do modelo:**
   - O C escreve os pesos em `model.bin` antes (ou logo após) imprimir `TRAINING_DONE`.
   - Isso garante que o arquivo `model.bin` estará íntegro quando o Python terminar de ler o stdout.

7. **Resumo do porquê do método:**
   - **Simplicidade:** apenas uso de pipes/stdout — nada de FFI, bindings, ou link dinâmico.
   - **Portabilidade:** funciona em qualquer sistema com Python e gcc (opcional).
   - **Isolamento:** o C executa nativamente e é responsável apenas pelo trabalho numérico; o Python cuida de UI/visualização.

---

## Formato da saída e do `model.bin` (explicação)

### Saída textual (stdout)
O `main.c` envia mensagens legíveis para que o controlador (Python) as interprete:
- `LOSS:<epoca>,<valor>` — a cada época (ou periodicamente) para permitir plot em tempo real.
- `TRAINING_DONE` — sinal de término; somente depois de gravar `model.bin`.

**Exemplo:**
```
LOSS:1,0.123456789
LOSS:2,0.098765432
...
TRAINING_DONE
```

### `model.bin`
- Formato: binário (o `main.c` escreve um cabeçalho simples seguido pelos floats dos pesos e biases).
- Uso: persistência para recarregar o modelo em execuções futuras sem re-treinar.
- Observação: o README não prescreve um formato byte-a-byte — consulte os comentários/rotinas `save_model` / `load_model` dentro de `main.c` para detalhes precisos (o código que você tem contém a rotina de leitura/gravação com verificação de “magic” e dimensões).

---

## Exemplo de execução e saída esperada

1. Executando:
```bash
make run
```

2. Saída no terminal (apenas um exemplo ilustrativo):
```
Compilando main.c -> central_comando
Iniciando processo: ./central_comando
LOSS:1,0.345678901
LOSS:2,0.234567890
LOSS:3,0.210987654
...
TRAINING_DONE
Processo finalizado com código 0
```

3. Janela do `matplotlib`:
- Gráfico em tempo real mostrando a curva de MSE decrescendo por época.
- Ao final, gráfico fixo com a curva completa.

---

## Soluções para problemas comuns / troubleshooting

- **`make: command not found` no Windows:**  
  - Use Git Bash, MSYS2, Cygwin, ou WSL; ou execute o `run_and_plot.py` diretamente com `python run_and_plot.py`.

- **Erro: `gcc: command not found`**  
  - Instale GCC (no Linux: `sudo apt install build-essential` / no Windows use MinGW/MSYS2 ou WSL).

- **`matplotlib` não instalado -> erro ao importar**  
  - Execute `pip install matplotlib` no venv ou no sistema.

- **O gráfico não atualiza em tempo real**  
  - Verifique se `run_and_plot.py` está lendo o `stdout` em modo texto e com `bufsize=1`. Se o C estiver bufferizando a saída, garanta que `fflush(stdout)` seja chamado no C após cada `printf` de progresso (o `main.c` enviado já tem `fflush` nos pontos certos).

- **`train.csv` mal formatado**  
  - Confirme número de colunas (10 por linha). Linhas incompletas podem ser puladas ou causar erro dependendo de como `main.c` trata `sscanf`. Veja logs de erro no terminal.

- **`model.bin` corrompido**  
  - Se uma execução terminou abruptamente (por falta de memória ou kill), apague `model.bin` e reexecute o treino.

---

## Exemplo final: passos recomendados para avaliação rápida
1. Clonar repositório:
```bash
git clone <https://github.com/GersonMenezes/RNparaClima>
cd <RNparaClima>
```
2. Criar virtualenv e instalar dependências:
```bash
python -m venv .venv
source .venv/bin/activate   # ou equivalente no Windows
pip install matplotlib numpy
```
3. Rodar:
```bash
make run
```
4. Se quiser limpar executáveis:
```bash
make clear
```

---
