#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define safetensors_file_size 548105171
#define params_raw_size 548090880 // saftensors_file_size - json_raw_size - 8
#define d_vocab  50257
#define raw_size 320827
#define enc_file_size 722883
#define token_file_size 676050
#define data_token_count 338025
#define json_raw_size 14283
#define d_model 768
#define d_seq 1024



struct token {
    uint32_t offset;
    uint32_t length;
};

struct decoder {
    struct token tokens[d_vocab ];
    char raw[raw_size];
};

struct data{
    uint16_t tokens[data_token_count];
};

struct params {
    struct {
        float* weight;
        float* bias;
    } wte;
        struct {
        float* weight;
        float* bias;
    } wpe;
    struct {
        struct {
            float* weight;
            float* bias;
        } ln_1;

        struct{
            struct  {
                float* weight;
                float* bias;
            } c_proj;

            struct  {
                float* weight;
                float* bias;
            } c_attn;
            
        } attn;
        struct {
            float* weight;
            float* bias;
        } ln_2;

        struct {
            struct {
                float* weight;
                float* bias;
            } c_fc;
                        struct {
                float* weight;
                float* bias;
            } c_proj;
        } mlp;
        
    } h[12];
    struct {
        float* weight;
        float* bias;
    } ln_f; 
};

struct activations {
    struct {
        float out[d_seq][d_model];
    } embedding;
    struct {
        struct{
            float out[d_seq][d_model];
        } ln_1;
    } h[12];
};

static size_t tensor_2_offset(char* json_raw, char* tensor_name, size_t expec_size) {
    // maybe end_offset is not needed
    // bcs 
    char *name_ptr = strstr(json_raw, tensor_name);
    assert(name_ptr);
    char* offset_ptr = strstr(name_ptr, "\"data_offsets\":[");
    assert(offset_ptr);
    offset_ptr = strchr(offset_ptr, '[');
    assert(offset_ptr);
    offset_ptr++;
    uint32_t start_offset = 0, end_offset = 0;
    sscanf(offset_ptr, "%u,%u", &start_offset, &end_offset);
    assert(end_offset - start_offset == expec_size * sizeof(float));
    return start_offset / 4; // 4 bytes per float so offset is in floats since we are using float* in params
}

int main() {
#define align(x) (((size_t)x + 255) & ~(size_t)0xff) // x has to have type size_t
    size_t offset = 0;
    size_t decoder_offset = offset;
    offset += align(sizeof(struct decoder));
    size_t data_offset = offset;
    offset += align(sizeof(struct data));
    size_t json_raw_offset = offset;
    offset += align(json_raw_size);
    size_t params_raw_offset = offset;
    offset += align(params_raw_size);
    size_t params_offset = offset;
    offset += align(sizeof(struct params));
    size_t activations_offset = offset;
    offset += align(sizeof(struct activations));
#undef align
    char* raw_mem = malloc(offset);
    struct decoder* decoder = (struct decoder*)(raw_mem + decoder_offset);
    struct data* data = (struct data*)(raw_mem + data_offset);
    char* json_raw = (char*)(raw_mem + json_raw_offset);
    float* params_raw = (float*)(raw_mem + params_raw_offset);
    struct params* params = (struct params*)(raw_mem + params_offset);
    struct activations* activations = (struct activations*)(raw_mem + activations_offset);
    {
        /* data */
    };
    

    {
    FILE* pf = fopen("data/enc", "r");
    assert(pf);
    unsigned long read = fread(decoder, 1, enc_file_size, pf);
    assert(read == enc_file_size);
    // for (int i = 0; i < d_vocab ; i++) {
    //     uint32_t o = decoder->tokens[i].offset;
    //     uint32_t l = decoder->tokens[i].length;
    //     printf("offset %u, size %u, %.*s\n", o, l, l, decoder->raw + o );
    // }
    fclose(pf);
    printf("test\n");
    }

    {

    FILE* pf = fopen("data/tokens", "r");
    assert(pf);
    unsigned long read = fread(data, 1, data_token_count * sizeof(uint16_t), pf);
    assert(read == data_token_count * sizeof(uint16_t));

    // for (int i = 0; i < data_token_count; i++) {
    //     uint16_t token = data->tokens[i];
    //     uint32_t offset = decoder->tokens[token].offset;
    //     uint32_t length = decoder->tokens[token].length;
    //     printf("%.*s", length, decoder->raw + offset);

        
    // }
    fclose(pf);
    }
    {
        FILE* pf = fopen("data/model.safetensors", "r");
        assert(pf);
        uint64_t json_size;
        unsigned long read = fread(&json_size, 1, 8, pf);
        assert(read == 8);
        
        read = fread(json_raw, 1, json_raw_size, pf);
        assert(read == json_raw_size);

        read = fread(params_raw, 1, params_raw_size, pf);
        assert(read == params_raw_size);

        params->wte.weight = params_raw + tensor_2_offset(json_raw, "wte.weight", d_vocab * d_model);
        params->wpe.weight = params_raw + tensor_2_offset(json_raw, "wpe.weight", d_seq* d_model);

        // here do the embedding 
        // TODO implement F.embedding
        // to produce wte_out and wpe_out
        // gives embd_out = wte_out + wpe_out

        

        for (int layer_i = 0; layer_i < 12; layer_i++) {
            char temp_name[64];

            // here check if first layer then ln_1_in = embd_out else ln_1_in = res_2_out (see python)

            // handle ln_1 weights and bias for each 12 layers
            printf("layer_i: %u\n", layer_i);
            sprintf(temp_name, "h.%u.ln_1.weight", layer_i);
            printf("temp_name: %s\n", temp_name);
            params->h[layer_i].ln_1.weight = params_raw + tensor_2_offset(json_raw, temp_name, d_model);

            sprintf(temp_name, "h.%u.ln_1.bias", layer_i);
            params->h[layer_i].ln_1.bias = params_raw + tensor_2_offset(json_raw, temp_name, d_model);

            // here do the layer norm to produce ln_1_out
            
            // handle attn weights and bias for each 12 layers
            sprintf(temp_name, "h.%u.attn.c_attn.weight", layer_i);
            params->h[layer_i].attn.c_attn.weight = params_raw + tensor_2_offset(json_raw, temp_name, d_model * 3 * d_model);

            sprintf(temp_name, "h.%u.attn.c_attn.bias", layer_i);
            params->h[layer_i].attn.c_attn.bias = params_raw + tensor_2_offset(json_raw, temp_name, d_model * 3);

            // here do the multihead attention calculations of q, k, v

            // attention projection 
            sprintf(temp_name, "h.%u.attn.c_proj.weight", layer_i);
            params->h[layer_i].attn.c_proj.weight = params_raw + tensor_2_offset(json_raw, temp_name, d_model * d_model);

            sprintf(temp_name, "h.%u.attn.c_proj.bias", layer_i);
            params->h[layer_i].attn.c_proj.bias = params_raw + tensor_2_offset(json_raw, temp_name, d_model);

            // here add residual conncection produce res_1_out

            // handle params for layer norm 2
            sprintf(temp_name, "h.%u.ln_2.weight", layer_i);
            params->h[layer_i].ln_2.weight = params_raw + tensor_2_offset(json_raw, temp_name, d_model);

            sprintf(temp_name, "h.%u.ln_2.bias", layer_i);
            params->h[layer_i].ln_2.bias = params_raw + tensor_2_offset(json_raw, temp_name, d_model);

            // here do the layer norm to produce ln_2_out

            // handle mlp weights and bias for each 12 layers
            sprintf(temp_name, "h.%u.mlp.c_fc.weight", layer_i);
            params->h[layer_i].mlp.c_fc.weight = params_raw + tensor_2_offset(json_raw, temp_name, d_model * 4 *d_model);

            sprintf(temp_name, "h.%u.mlp.c_fc.bias", layer_i);
            params->h[layer_i].mlp.c_fc.bias = params_raw + tensor_2_offset(json_raw, temp_name, d_model * 4);

            // here do the linear transformation

            // here do gelu activation

            // handle mlp projection weights and bias for each 12 layers
            sprintf(temp_name, "h.%u.mlp.c_proj.weight", layer_i);
            params->h[layer_i].mlp.c_proj.weight = params_raw + tensor_2_offset(json_raw, temp_name, 4 * d_model * d_model);

            sprintf(temp_name, "h.%u.mlp.c_proj.bias", layer_i);
            params->h[layer_i].mlp.c_proj.bias = params_raw + tensor_2_offset(json_raw, temp_name, d_model);

            // linear transformation

            // here add residual conncection
            // res_2_out = res_1_out + mlp_c_proj_out
        }

        // handle params for layer norm f
       

        params->ln_f.weight = params_raw + tensor_2_offset(json_raw, "ln_f.weight", d_model);
        params->ln_f.bias = params_raw + tensor_2_offset(json_raw, "ln_f.bias", d_model);

        // here do the layer norm to produce ln_f_out

        // here do the unembedding
        // TODO implement F.linear

        // argmax to get the token
        // decode token index

        // compute cross entropy loss with unembedding and target


        fclose(pf);
    }
    size_t inp_size = 64;
    uint16_t* inp = data->tokens;

    for (size_t i = 0; i < inp_size; i++) {
        uint16_t token = inp[i];

        float* wte = params->wte.weight + token * d_model;

        float* wpe = params->wpe.weight + i * d_model;

        float* out = (float*)activations->embedding.out + i * d_model;
        float* out_end = out + d_model;

        for (; out != out_end; out++, wte++, wpe++) {
            *out = *wte + *wpe;
        }

    }
    double sum = 0.0;
    for (size_t i = 0; i < inp_size * d_model; i++) {
    sum += (double)((float*)activations->embedding.out)[i];
    }
    printf("sum: %f\n", sum);

    int layer_idx = 0;

    for (size_t i = 0; i < inp_size + d_model; i++) {
        float* in = (float*)activations->embedding.out + i * d_model;
        float* in_end = in + d_model;
        float* in_reset = in;

        float mean = 0.0f;
        for (; in != in_end; in++) {
            mean += *in;
            }
        mean /= d_model;

        float SSD = 0.0f;
        for (in = in_reset; in != in_end; in++) {
            float diff = *in - mean;
            SSD += diff * diff;
        }

        float inv_std = 1.0f / sqrt(SSD / d_model);

        float* out = (float*)activations->h[layer_idx].ln_1.out + i * d_model;
        float* weight = params->h[layer_idx].ln_1.weight;
        float* bias = params->h[layer_idx].ln_1.bias;
        // normalize
        for (in = in_reset; in != in_end; in++, weight++, bias++, out++) {
            float in_norm = (*in - mean) * inv_std;
            *out = in_norm * *weight + *bias;
        }
    }
    double sum = 0.0;
    for (size_t i = 0; i < inp_size * d_model; i++) {
        sum += (double)((float*)activations->h[layer_idx].out)[i];
    }   

    
    return 0;
}

