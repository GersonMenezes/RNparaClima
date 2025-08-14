// main.c  (salva/carrega pesos em model.bin; modo --load-model para testar)
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include <string.h>

#define INPUT_NODES 8
#define OUTPUT_NODES 2
#define HIDDEN_NODES 16
#define EPOCHS 100
#define LEARNING_RATE 0.001f
#define MAX_DATA 100000
#define SHUFFLE 1

// Funções utilitárias inline para performance
static inline float relu(float x){ return x > 0 ? x : 0; }
static inline float drelu(float y){ return y > 0 ? 1 : 0; }
static inline float zscore(float x, float mean, float std){ return (x - mean) / (std + 1e-8f); }
static inline float inv_zscore(float z, float mean, float std){ return z * std + mean; }
static inline float rand_weight(){ return ((float)rand() / RAND_MAX) * 2.0f - 1.0f; }

// Estrutura da Rede Neural
typedef struct {
    float input[INPUT_NODES],
          hidden[HIDDEN_NODES],
          output[OUTPUT_NODES];

    float weights_ih[INPUT_NODES][HIDDEN_NODES],
          weights_ho[HIDDEN_NODES][OUTPUT_NODES];

    float bias_h[HIDDEN_NODES],
          bias_o[OUTPUT_NODES];
} NeuralNetwork;

// Estrutura para estatísticas (média e desvio padrão)
typedef struct { float mean, std; } Stats;
static Stats in_stats[INPUT_NODES], out_stats[OUTPUT_NODES];

/* ---------- Salvar / Carregar modelo ---------- */
int save_model(const char *filename, NeuralNetwork *nn) {
    FILE *f = fopen(filename, "wb");
    if (!f) return 0;
    fwrite("NNMD", 1, 4, f); // Magic number
    int dims[3] = {INPUT_NODES, HIDDEN_NODES, OUTPUT_NODES};
    fwrite(dims, sizeof(int), 3, f);
    fwrite(nn->weights_ih, sizeof(float), INPUT_NODES * HIDDEN_NODES, f);
    fwrite(nn->weights_ho, sizeof(float), HIDDEN_NODES * OUTPUT_NODES, f);
    fwrite(nn->bias_h, sizeof(float), HIDDEN_NODES, f);
    fwrite(nn->bias_o, sizeof(float), OUTPUT_NODES, f);
    fwrite(in_stats, sizeof(Stats), INPUT_NODES, f);
    fwrite(out_stats, sizeof(Stats), OUTPUT_NODES, f);
    fclose(f);
    return 1;
}

int load_model(const char *filename, NeuralNetwork *nn) {
    FILE *f = fopen(filename, "rb");
    if (!f) return 0;
    char magic[5] = {0};
    fread(magic, 1, 4, f);
    if (strncmp(magic,"NNMD",4) != 0) { fclose(f); return 0; }
    int dims[3];
    fread(dims, sizeof(int), 3, f);
    if (dims[0] != INPUT_NODES || dims[1] != HIDDEN_NODES || dims[2] != OUTPUT_NODES) { fclose(f); return 0; }
    fread(nn->weights_ih, sizeof(float), INPUT_NODES * HIDDEN_NODES, f);
    fread(nn->weights_ho, sizeof(float), HIDDEN_NODES * OUTPUT_NODES, f);
    fread(nn->bias_h, sizeof(float), HIDDEN_NODES, f);
    fread(nn->bias_o, sizeof(float), OUTPUT_NODES, f);
    fread(in_stats, sizeof(Stats), INPUT_NODES, f);
    fread(out_stats, sizeof(Stats), OUTPUT_NODES, f);
    fclose(f);
    return 1;
}

/* ---------- Rede e treinamento ---------- */
// Inicializa a rede com pesos aleatórios
void init_network(NeuralNetwork *nn) {
    for (int i = 0; i < INPUT_NODES; ++i)
        for (int j = 0; j < HIDDEN_NODES; ++j)
            nn->weights_ih[i][j] = rand_weight();

    for (int i = 0; i < HIDDEN_NODES; ++i) {
        nn->bias_h[i] = rand_weight();
        for (int j = 0; j < OUTPUT_NODES; ++j)
            nn->weights_ho[i][j] = rand_weight();
    }
    for (int i = 0; i < OUTPUT_NODES; ++i)
        nn->bias_o[i] = rand_weight();
}

// Carrega os dados de um arquivo CSV
int load_csv(const char* filename, float x[][INPUT_NODES], float y[][OUTPUT_NODES]) {
    FILE* f = fopen(filename, "r");
    if (!f) return 0;
    char line[256];
    fgets(line, sizeof(line), f); // pular header
    int count = 0;
    while (fgets(line, sizeof(line), f) && count < MAX_DATA) {
        sscanf(line, "%f,%f,%f,%f,%f,%f,%f,%f,%f,%f",
               &x[count][0], &x[count][1], &x[count][2], &x[count][3],
               &x[count][4], &x[count][5], &x[count][6], &x[count][7],
               &y[count][0], &y[count][1]);
        count++;
    }
    fclose(f);
    return count;
}

// Calcula as estatísticas (média/desvio padrão) para normalização
void calc_stats(float x[][INPUT_NODES], float y[][OUTPUT_NODES], int n) {
    for (int j = 0; j < INPUT_NODES; ++j) {
        double sum = 0;
        for (int i = 0; i < n; ++i) sum += x[i][j];
        in_stats[j].mean = sum / n;
        double var = 0;
        for (int i = 0; i < n; ++i) var += pow(x[i][j] - in_stats[j].mean, 2);
        in_stats[j].std = sqrt(var / n);
    }
    for (int j = 0; j < OUTPUT_NODES; ++j) {
        double sum = 0;
        for (int i = 0; i < n; ++i) sum += y[i][j];
        out_stats[j].mean = sum / n;
        double var = 0;
        for (int i = 0; i < n; ++i) var += pow(y[i][j] - out_stats[j].mean, 2);
        out_stats[j].std = sqrt(var / n);
    }
}

// Roda a propagação direta (feedforward)
void feedforward(NeuralNetwork *nn) {
    for (int i = 0; i < HIDDEN_NODES; ++i) {
        float sum = nn->bias_h[i];
        for (int j = 0; j < INPUT_NODES; ++j) sum += nn->input[j] * nn->weights_ih[j][i];
        nn->hidden[i] = relu(sum);
    }
    for (int i = 0; i < OUTPUT_NODES; ++i) {
        float sum = nn->bias_o[i];
        for (int j = 0; j < HIDDEN_NODES; ++j) sum += nn->hidden[j] * nn->weights_ho[j][i];
        nn->output[i] = sum;
    }
}

// Roda uma passada de treinamento (feedforward + backpropagation)
void train(NeuralNetwork *nn, float input[], float target[]) {
    // A função de treino já executa o feedforward.
    memcpy(nn->input, input, sizeof(float)*INPUT_NODES);
    feedforward(nn);
    
    // Calcula o erro na saída
    float err_o[OUTPUT_NODES];
    for (int i = 0; i < OUTPUT_NODES; ++i) err_o[i] = target[i] - nn->output[i];
    
    // Ajusta pesos da camada oculta para a saída
    for (int i = 0; i < OUTPUT_NODES; ++i) {
        float grad = err_o[i] * LEARNING_RATE;
        nn->bias_o[i] += grad;
        for (int j = 0; j < HIDDEN_NODES; ++j) nn->weights_ho[j][i] += grad * nn->hidden[j];
    }
    
    // Calcula o erro na camada oculta
    float err_h[HIDDEN_NODES];
    for (int i = 0; i < HIDDEN_NODES; ++i) err_h[i] = 0.0f;
    for (int i = 0; i < HIDDEN_NODES; ++i)
        for (int j = 0; j < OUTPUT_NODES; ++j) err_h[i] += nn->weights_ho[i][j] * err_o[j];
    
    // Ajusta pesos da camada de entrada para a oculta
    for (int i = 0; i < HIDDEN_NODES; ++i) {
        float grad = drelu(nn->hidden[i]) * err_h[i] * LEARNING_RATE;
        nn->bias_h[i] += grad;
        for (int j = 0; j < INPUT_NODES; ++j) nn->weights_ih[j][i] += grad * nn->input[j];
    }
}

// Embaralha os índices dos dados
void shuffle_indices(int *idx, int n) {
    for (int i = n-1; i > 0; --i){
        int j = rand() % (i+1);
        int tmp = idx[i]; idx[i] = idx[j]; idx[j] = tmp;
    }
}

int main(int argc, char **argv) {
    int no_interactive = 0;
    int load_model_flag = 0;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--no-interactive") == 0 || strcmp(argv[i], "-n") == 0) no_interactive = 1;
        if (strcmp(argv[i], "--load-model") == 0) load_model_flag = 1;
    }

    srand((unsigned)time(NULL));
    NeuralNetwork nn;

    if (load_model_flag) {
        if (!load_model("model.bin", &nn)) {
            printf("Erro: model.bin não encontrado ou corrompido.\n");
            return 1;
        }
        printf("Modelo carregado de model.bin. Indo para modo interativo.\n");
    } else {
        init_network(&nn);

        static float x[MAX_DATA][INPUT_NODES], y[MAX_DATA][OUTPUT_NODES];
        int n = load_csv("train.csv", x, y);
        if (!n) { printf("Erro ao ler train.csv\n"); return 1; }

        calc_stats(x, y, n);

        static float x_norm[MAX_DATA][INPUT_NODES], y_norm[MAX_DATA][OUTPUT_NODES];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < INPUT_NODES; ++j) x_norm[i][j] = zscore(x[i][j], in_stats[j].mean, in_stats[j].std);
            for (int j = 0; j < OUTPUT_NODES; ++j) y_norm[i][j] = zscore(y[i][j], out_stats[j].mean, out_stats[j].std);
        }

        int indices[MAX_DATA];
        for (int i = 0; i < n; ++i) indices[i] = i;

        // Loop de treinamento
        for (int e = 0; e < EPOCHS; ++e) {
            if (SHUFFLE) shuffle_indices(indices, n);
            double epoch_loss = 0.0;
            for (int k = 0; k < n; ++k) {
                // ### LÓGICA ORIGINAL RESTAURADA AQUI ###
                // Esta é a forma como o seu código calculava a perda, que estava correta.
                memcpy(nn.input, x_norm[indices[k]], sizeof(nn.input));
                feedforward(&nn);
                float err0 = y_norm[indices[k]][0] - nn.output[0];
                float err1 = y_norm[indices[k]][1] - nn.output[1];
                epoch_loss += err0*err0 + err1*err1;
                train(&nn, x_norm[indices[k]], y_norm[indices[k]]);
            }
            double mse = epoch_loss / (n * OUTPUT_NODES);
            printf("Epoch %3d/%d  -  Loss (MSE): %.9f\n", e+1, EPOCHS, mse);
            printf("LOSS:%d,%.9f\n", e+1, mse);
            fflush(stdout);
        }

        // salvar modelo ao final do treinamento
        if (save_model("model.bin", &nn)) {
            printf("Modelo salvo em model.bin\n");
        } else {
            printf("Falha ao salvar model.bin\n");
        }

        printf("TRAINING_DONE\n");
        fflush(stdout);

        if (no_interactive) return 0;
    }

    // modo interativo
    
    const char* prompts[INPUT_NODES] = {"Temperatura", "Umidade", "Vento", "Hora", "Dia", "Nuvens", "Pressao", "Precipitacao"};
    float user_in[INPUT_NODES];
    for (int j = 0; j < INPUT_NODES; ++j) {
        printf("%s: ", prompts[j]);
        if (scanf("%f", &user_in[j]) != 1) {
            int c; while ((c = getchar()) != '\n' && c != EOF);
            j--; continue;
        }
    }
    float norm_in[INPUT_NODES];
    for (int j = 0; j < INPUT_NODES; ++j) norm_in[j] = zscore(user_in[j], in_stats[j].mean, in_stats[j].std);
    memcpy(nn.input, norm_in, sizeof(norm_in));
    feedforward(&nn);
    float out0 = inv_zscore(nn.output[0], out_stats[0].mean, out_stats[0].std);
    float out1 = inv_zscore(nn.output[1], out_stats[1].mean, out_stats[1].std);
    printf("\n ---------------Saída------------------\n");
    printf("Sensacao: %.2fC, Probabilidade de Chuva: %.2f%%\n\n", out0, out1);


    return 0;
}