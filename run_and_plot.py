# -*- coding: utf-8 -*-
"""
@file run_and_plot.py
@author Gerson Menezes, Eduardo Timm
@brief Script Python para orquestrar e visualizar o treinamento da Rede Neural em C.
@version 1.0
@date 2025-08-14

@copyright Copyright (c) 2025

@details
Este script atua como o "centro de comando" para a aplicação. Ele é responsável
por gerenciar o ciclo de vida do programa em C e fornecer uma visualização
em tempo real do processo de treinamento. Suas tarefas incluem:
- Verificar se o executável C ('central_comando') existe e compilá-lo se necessário.
- Iniciar o programa C como um subprocesso, passando a flag '--no-interactive'.
- Capturar a saída padrão (stdout) do subprocesso C em uma thread separada para
  não bloquear a interface gráfica.
- Interpretar (parse) as mensagens de progresso ('LOSS:epoca,mse') enviadas pelo C.
- Utilizar Matplotlib para plotar um gráfico animado da perda (MSE) por época.
- Gerenciar o encerramento seguro do subprocesso C quando o gráfico é fechado.
A comunicação é feita via pipes de stdout, uma forma de Comunicação entre Processos (IPC).
"""

# run_and_plot.py
import os
import sys
import subprocess
import threading
import platform
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from threading import Thread, Lock

# Tenta forçar o uso do backend 'TkAgg' para o Matplotlib, que é um backend
# de interface gráfica interativa. Isso aumenta a compatibilidade em diferentes
# sistemas operacionais. Se falhar, o programa continua com o backend padrão.
try:
    matplotlib.use('TkAgg')
except Exception:
    pass

# --- 1. CONFIGURAÇÃO E COMPILAÇÃO DO CÓDIGO C ---

# Define o nome do arquivo fonte em C.
SRC = "main.c"
# ### NOVO (Melhoria) ###
# Define o número de épocas para configurar o gráfico (deve ser o mesmo do main.c).
TOTAL_EPOCHS = 100

# Define o nome do arquivo binário (executável) e o comando de compilação
# de acordo com o sistema operacional (Windows ou outro, como Linux/macOS).
if platform.system() == "Windows":
    BIN = "central_comando.exe"
    COMPILE_CMD = ["gcc", SRC, "-o", BIN, "-lm"]
else:
    BIN = "./central_comando"
    COMPILE_CMD = ["gcc", SRC, "-o", "central_comando", "-lm"]

# Verifica se o arquivo executável já existe.
if not os.path.exists(BIN):
    print("Binário não encontrado. Tentando compilar:", " ".join(COMPILE_CMD))
    # Se não existir, executa o comando de compilação usando o GCC.
    # A saída de erro (stderr) e padrão (stdout) são capturadas.
    r = subprocess.run(COMPILE_CMD, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    
    # Se o código de retorno for diferente de 0, significa que houve um erro na compilação.
    if r.returncode != 0:
        print("Erro ao compilar:\n", r.stderr)
        sys.exit(1) # Encerra o script com um código de erro.
    print("Compilado com sucesso.")

print("Matplotlib backend:", matplotlib.get_backend())

# --- 2. PREPARAÇÃO PARA COMUNicação E PLOTAGEM ---

# Listas para armazenar os dados recebidos do programa C.
epochs = []
mses = []
# 'Lock' é um mecanismo de sincronização para evitar que a thread de leitura
# e a thread principal (de plotagem) acessem as listas ao mesmo tempo (condição de corrida).
lock = Lock()

# Variável para controlar o estado do treinamento.
# Inicia como False e será alterada para True quando o C avisar que terminou.
training_finished = False

# Define o comando para executar o programa C.
cmd = [BIN, "--no-interactive"]

# Inicia o processo C.
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

# --- 3. THREAD PARA LEITURA DA SAÍDA DO PROCESSO C ---

# Esta função será executada em uma thread separada para ler a saída do C
def reader_thread(p):
    # Declara que vamos modificar a variável global 'training_finished'.
    global training_finished
    
    # Itera sobre cada linha que o processo C imprime (via stdout).
    for raw in p.stdout:
        line = raw.strip() # Remove espaços em branco e quebras de linha.
        print("[C STDOUT]", line) # Mostra no console do Python tudo que o C envia.

        if line.startswith("LOSS:"):
            try:
                payload = line[5:]
                e_str, m_str = payload.split(",", 1) # Divide a string em época e MSE.
                e = int(e_str)
                m = float(m_str)
                
                with lock:
                    epochs.append(e)
                    mses.append(m)
                print(f"[Parsed] epoch={e} mse={m:.9f}")
            except Exception as ex:
                print("[Parse error]", ex, "line:", line)
        
        # Verifica se o treinamento no C terminou.
        elif line.startswith("TRAINING_DONE"):
            print("[C] Treinamento finalizado.\nFeche a janela do gráfico para encerrar o script.")
            # Ao receber o sinal, adquire o lock e atualiza a variável de estado.
            with lock:
                training_finished = True
    
    # Fecha o canal de comunicação quando o processo C terminar.
    p.stdout.close()

# Cria e inicia a thread de leitura.
t = Thread(target=reader_thread, args=(proc,), daemon=True)
t.start()

# --- 4. CONFIGURAÇÃO E EXECUÇÃO DA PLOTAGEM AO VIVO ---

fig, ax = plt.subplots()
line, = ax.plot([], [], marker='o')

# Configurações visuais do gráfico.
ax.set_xlabel("Época")
ax.set_ylabel("MSE")

# Define o título inicial com a cor verde.
title_obj = ax.set_title("Loss (MSE) - Running", color='green')
ax.grid(True)

# ### NOVO (Melhoria) ###
# Define os limites iniciais dos eixos para evitar avisos (warnings).
ax.set_xlim(0, TOTAL_EPOCHS + 1)
ax.set_ylim(0, 0.01)

def init():
    line.set_data([], [])
    return line,

# Função que será chamada periodicamente para atualizar o gráfico.
def update(frame):
    # Adquire o lock para ler os dados e o estado do treinamento de forma segura.
    with lock:
        xs = list(epochs)
        ys = list(mses)
        is_finished = training_finished
    
    # Se não houver dados, não faz nada.
    if not xs:
        # Mas se não houver dados E já tiver terminado, para a animação.
        if is_finished:
             if ani.event_source is not None:
                ani.event_source.stop()
        return line,
    
    # Atualiza os dados (x, y) da linha do gráfico.
    line.set_data(xs, ys)

    # Reajusta automaticamente a escala do eixo Y para se adequar aos novos valores de MSE.
    ax.relim()
    ax.autoscale_view(scalex=False, scaley=True)

    # ### MODIFICADO ###
    # Lógica para finalizar a animação (sem fechar a janela).
    if is_finished:
        # Muda o título para "Finished" e a cor para vermelho.
        ax.set_title("Loss (MSE) - Finished", color='red')
        
        # Para a animação para de chamar esta função repetidamente.
        # O gráfico permanecerá visível até que o usuário o feche.
        if ani.event_source is not None:
             ani.event_source.stop()

    return line,

# Cria o objeto de animação.
ani = FuncAnimation(fig, update, init_func=init, interval=200, cache_frame_data=False)

try:
    # Mostra a janela do gráfico. Esta chamada é bloqueante e só termina
    # quando a janela do gráfico é fechada manualmente.
    plt.show()
finally:
    # Este bloco é executado quando a janela é fechada.
    # Garante que o processo C seja encerrado.
    if proc.poll() is None:
        print("Fechando processo C...")
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            proc.kill()
    
    # Imprime a mensagem final após a janela ser fechada.
    print("Script Python finalizado.")