#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <fcntl.h>

// Define the configuration for the MLP
typedef struct {
    int input_dim;   // Input dimension
    int hidden_dim;  // Hidden layer dimension
    int output_dim;  // Output dimension
} Config;

// Define the weights for the MLP
typedef struct {
    float *w1;  // Weights for the first layer
    float *b1;  // Biases for the first layer
    float *w2;  // Weights for the second layer
    float *b2;  // Biases for the second layer
} MLPWeights;

// Define the run state for the MLP
typedef struct {
    float *input;    // Input buffer
    float *hidden;  // Buffer for the first hidden layer
    float *output;   // Output buffer
} RunState;

// Define the MLP structure
typedef struct {
    Config config;        // The configuration of the MLP
    MLPWeights weights;   // The weights of the MLP
    RunState state;       // Buffers for the forward pass
    // some more state needed to properly clean up the memory mapping (sigh)
    int fd; // file descriptor for memory mapping
    float* data; // memory mapped data pointer
    ssize_t file_size; // size of the checkpoint file in bytes
} MLP;

void malloc_run_state(RunState* s, Config* p) {
    s->input = calloc(p->input_dim, sizeof(float));
    s->hidden = calloc(p->hidden_dim, sizeof(float));
    s->output = calloc(p->output_dim, sizeof(float));
    // ensure all mallocs went fine
    if (!s->input || !s->hidden || !s->output) 
    {
        fprintf(stderr, "malloc failed!\n");
        exit(EXIT_FAILURE);
    }
}

void free_run_state(RunState* s) {
    free(s->input);
    free(s->hidden);
    free(s->output);
}

void memory_map_weights(MLPWeights *w, Config* p, float* ptr) {
    w->w1 = ptr;
    ptr += p->input_dim * p->hidden_dim;

    w->b1 = ptr;
    ptr += p->hidden_dim;

    w->w2 = ptr;
    ptr += p->output_dim * p->hidden_dim;

    w->b2 = ptr;
    ptr += p->output_dim;
}

void read_checkpoint(char* checkpoint, Config* config, MLPWeights* weights,
                     int* fd, float** data, ssize_t* file_size) {
    FILE *file = fopen(checkpoint, "rb");
    if (!file) { fprintf(stderr, "Couldn't open file %s\n", checkpoint); exit(EXIT_FAILURE); }
    // read in the config header
    if (fread(config, sizeof(Config), 1, file) != 1) { exit(EXIT_FAILURE); }

    // figure out the file size
    fseek(file, 0, SEEK_END); // move file pointer to end of file
    *file_size = ftell(file); // get the file size, in bytes
    fclose(file);

    // memory map the MLP weights into the data pointer
    *fd = open(checkpoint, O_RDONLY); // open in read only mode
    if (*fd == -1) { fprintf(stderr, "open failed!\n"); exit(EXIT_FAILURE); }
    *data = mmap(NULL, *file_size, PROT_READ, MAP_PRIVATE, *fd, 0);
    if (*data == MAP_FAILED) { fprintf(stderr, "mmap failed!\n"); exit(EXIT_FAILURE); }
    float* weights_ptr = *data + sizeof(Config)/sizeof(float);
    memory_map_weights(weights, config, weights_ptr);
}

void build_mlp(MLP *t, char* checkpoint_path) {
    // read in the Config and the Weights from the checkpoint
    read_checkpoint(checkpoint_path, &t->config, &t->weights, &t->fd, &t->data, &t->file_size);
    // allocate the RunState buffers
    malloc_run_state(&t->state, &t->config);
}

void free_mlp(MLP* t) {
    // close the memory mapping
    if (t->data != MAP_FAILED) { munmap(t->data, t->file_size); }
    if (t->fd != -1) { close(t->fd); }
    // free the RunState buffers
    free_run_state(&t->state);
}

void softmax(float* x, int size) {
    // find max value (for numerical stability)
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }
    // exp and sum
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    // normalize
    for (int i = 0; i < size; i++) {
        x[i] /= sum;
    }
}

void matmul(float* xout, float* x, float* w, int n, int d) {
    // W (d,n) @ x (n,) -> xout (d,)
    // by far the most amount of time is spent inside this little function
    int i;
    #pragma omp parallel for private(i)
    for (i = 0; i < d; i++) {
        float val = 0.0f;
        for (int j = 0; j < n; j++) {
            val += w[i * n + j] * x[j];
        }
        xout[i] = val;
    }
}

void relu(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = fmaxf(0, x[i]);  // Apply ReLU activation element-wise
    }
}

void add_bias(float* x, float* b, int size) {
    for (int i = 0; i < size; i++) {
        x[i] += b[i];  // Add bias element-wise
    }
}
float* forward(MLP* mlp, int input, char mode) {

    // Layer 1: Linear -> ReLU
    matmul(mlp->state.hidden, input, mlp->weights.w1, mlp->config.input_dim, mlp->config.hidden_dim);
    add_bias(mlp->state.hidden, mlp->weights.b1, mlp->config.hidden_dim);
    relu(mlp->state.hidden, mlp->config.hidden_dim);

    // Layer 3: Linear -> Softmax
    matmul(mlp->state.output, mlp->state.hidden, mlp->weights.w2, mlp->config.hidden_dim, mlp->config.output_dim);
    add_bias(mlp->state.output, mlp->weights.b2, mlp->config.output_dim);
    softmax(mlp->state.output, mlp->config.output_dim);
}

int main(int argc, char *argv[]) {

    // default parameters
    char *checkpoint_path = NULL;  // e.g. out/model.bin
    char *prompt = NULL;        // prompt string
    unsigned long long rng_seed = 0; // seed rng with time by default
    char *mode = "generate";    // infer|train

    // poor man's C argparse so we can override the defaults above from the command line
    if (argc >= 2) { checkpoint_path = argv[1]; } else { error_usage(); }
    for (int i = 2; i < argc; i+=2) {
        // do some basic validation
        if (i + 1 >= argc) { error_usage(); } // must have arg after flag
        if (argv[i][0] != '-') { error_usage(); } // must start with dash
        if (strlen(argv[i]) != 2) { error_usage(); } // must be -x (one dash, one letter)
        // read in the args
        if (argv[i][1] == 's') { rng_seed = atoi(argv[i + 1]); }
        else if (argv[i][1] == 'i') { prompt = argv[i + 1]; }
        else if (argv[i][1] == 'm') { mode = argv[i + 1]; }
        else { error_usage(); }
    }

    // parameter validation/overrides
    if (rng_seed <= 0) rng_seed = (unsigned int)time(NULL);

    // build the Transformer via the model .bin file
    MLP mlp;
    build_mlp(&mlp, checkpoint_path);

    // run!
    run(&mlp, prompt, mode);

    // memory cleanup
    free_mlp(&mlp);
    return 0;
}


