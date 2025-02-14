#include <stdio.h>
#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
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
#define d_k 64
#define inv_sqrt_k 0.125f
#define n_layers 12


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
            float mean[d_seq];
            float inv_std[d_seq];
            float out[d_seq][d_model];
        } ln_1;
        struct {
            struct {
                float out[d_seq][3 * d_model];
            } c_attn;
            struct {
                float out[12][d_seq][d_seq];
            } attn;
            struct {
                float out[12][d_seq][d_seq];
            } s;
            struct {
                float out[d_seq][d_model];
            } z; 
            struct {
                float out[d_seq][d_model];
            } c_proj;           
        } attn;
        struct {
            float out[d_seq][d_model];
        } res_1;
        struct {
            float mean[d_seq];
            float inv_std[d_seq];
            float out[d_seq][d_model];
        } ln_2;
        struct {
            struct {
                float out[d_seq][4 * d_model];
            } c_fc;
            struct {
                float out[d_seq][4 * d_model];
            } gelu;
            struct {
                float out[d_seq][d_model];
            } c_proj;
        } mlp;
        struct {
            float out[d_seq][d_model];
        } res_2;
    } h[12];
    struct {
        float mean[d_seq];
        float inv_std[d_seq];
        float out[d_seq][d_model];
    } ln_f;
    struct {
        float out[d_seq][d_vocab];
    } unembedding;
    struct {
        float out[d_seq][d_vocab];
    } prob;
};

struct grads{
    struct {
        struct {
            struct {
                float weight[d_model][4 * d_model];
                float bias[4 * d_model];
            } c_fc;
            struct {
                float weight[4 * d_model][d_model];
                float bias[d_model];
            } c_proj;                   
        } mlp;
    } h[12];

    struct {
        float weight[d_model];
        float bias[d_model];
    } ln_f;
    struct {
        float weight[d_vocab][d_model];
    } wte;
};

struct activations_back {
    struct {
        struct{
            struct {
                float out[d_seq][4 * d_model];
            } c_fc;
            struct {
                float out[d_seq][4 * d_model];
            } gelu;
        } mlp;
        struct {
            float out[d_seq][d_model];
        } res_2;
    } h[12];
    struct {
        float out[d_seq][d_model];
    } ln_f;
    struct {
        float out[d_seq][d_vocab];
    } unembedding;

};

struct linear {
    const float* in;
    const float* weights;
    const float* bias;
    float* out;
    size_t in_size;
    size_t out_size;
    size_t sample_size; 
};

struct ln {
    const float* in;
    const float* weight;
    const float* bias;
    float* inv_std;
    float* mean;
    float* out;
    size_t in_size; // # elements per row
    size_t sample_size;
};

struct ln_back {
    const float* in;
    const float* weight;
    const float* bias;
    const float* dl_dout;
    float* inv_std;
    float* mean;
    float* dl_din;
    float* dl_dweight;
    float* dl_dbias;
    size_t in_size; // # elements per row
    size_t sample_size;
};

struct linear_back {
    const float* in;
    const float* weight;
    const float* bias;
    const float* dl_dout;
    float* dl_dweight;
    float* dl_dbias;
    size_t in_size;
    size_t out_size;
    size_t sample_size;
};

static void layer_norm(const struct ln* ln) {
    for (size_t i = 0; i < ln->sample_size; i++) {
        const float* in = ln->in + i * ln->in_size;
        const float* in_end = in + ln->in_size;
        const float* in_reset = in;

        float mean = 0.0f;
        for (; in != in_end; in++) {
            mean += *in;
            }
        mean /= ln->in_size;
        ln->mean[i] = mean;

        float SSD = 0.0f;
        for (in = in_reset; in != in_end; in++) {
            float diff = *in - mean;
            SSD += diff * diff;
        }

        float inv_std = 1.0f / sqrtf(SSD / ln->in_size);
        ln->inv_std[i] = inv_std;

        const float* weight = ln->weight;
        const float* bias = ln->bias;
        float* out = ln->out + i * ln->in_size;
        // normalize
        for (in = in_reset; in != in_end; in++, weight++, bias++, out++) {
            float in_norm = (*in - mean) * inv_std;
            *out = in_norm * *weight + *bias;
        }
    }
}



static void linear(const struct linear_back* lin) {
    for (size_t i = 0; i < lin->sample_size; i++) {
        const float* in = lin->in + i * lin->in_size;
        const float* weight = lin->weight;
        const float* weight_end = weight + lin->in_size * lin->out_size;
        const float* bias = lin->bias;

        float* out = lin->out + i * lin->out_size;
        float* out_end = out + lin->out_size;
        float* out_reset = out;

        memcpy(out, bias, lin->out_size * sizeof(float));

        while (true) {
            *out += *weight * *in;
            weight++;
            out++;
            if (out == out_end) {
                out = out_reset;
                in++;
                if (weight == weight_end) {
                    break;
                }
            }
        }
    }
}

static void layer_norm_back (const struct ln_back* ln) {
    for (size_t i = 0; i < ln->sample_size; i++) {
        float* dl_din = ln->dl_din + i * ln->in_size;
        const float* in = ln->in + i * ln->in_size;
        const float* in_reset = in;
        
        const float* dl_dout = ln->dl_dout + i * ln->in_size;
        const float* dl_dout_end = dl_dout + ln->in_size;
        const float* dl_dout_reset = dl_dout;

        float* dl_dbias = ln->dl_dbias;
        float* dl_dweight = ln->dl_dweight;
        const float* weight = ln->weight;
        const float* weight_reset = weight;

        float mean = ln->mean[i];
        float inv_std = ln->inv_std[i];

        float mean_partial = 0.0f;
        float std_partial = 0.0f;

        for (; dl_dout != dl_dout_end; dl_dout++, dl_dbias++, dl_dweight++, weight++, in++) {
            float dl_din_norm = *dl_dout * *weight; // dL/din_norm
            float in_norm = (*in - mean) * inv_std; // in_norm
            *dl_dbias += *dl_dout;
            *dl_dweight += *dl_dout * in_norm;

            mean_partial += dl_din_norm;
            std_partial += dl_din_norm * in_norm;
        }

        mean_partial /= (float)ln->in_size;
        std_partial /= (float)ln->in_size;

        dl_dout = dl_dout_reset;
        in = in_reset;
        weight = weight_reset;

        for (; dl_dout != dl_dout_end; dl_dout++, in++, dl_din++, weight++) {
            float dl_din_norm = *dl_dout * *weight;
            float in_norm = (*in - mean) * inv_std;

            float dl_din_ = 0.0f;
            dl_din_ += dl_din_norm;
            dl_din_ -= mean_partial;
            dl_din_ -= std_partial * in_norm;
            dl_din_ *= inv_std;
            *dl_din = dl_din_;
        }
    }
}

static void linear_back(const struct linear* lin) {
    for (size_t i = 0; i < lin->in_size; i++){
        float* dl_dweight = lin->weight;
        float* dl_dweight_end = dl_dweight + 4 * d_model * d_model;
        float* dl_dbias = (float*)grads->h[11].mlp.c_proj.bias;
        const float* dl_dout = (float*)activations_back->h[11].res_2.out + i * d_model;
        const float* dl_dout_end = dl_dout + d_model;
        const float* dl_dout_reset = dl_dout;

        const float* in = (float*)activations->h[11].mlp.gelu.out + i * 4 * d_model;

        for (; dl_dout != dl_dout_end; dl_dout++, dl_dbias++) { //  dl_dweight++,
            *dl_dbias += *dl_dout;
        }

        dl_dout = dl_dout_reset;

        while (true) {
            *dl_dweight += *in * *dl_dout;
            dl_dout++;
            dl_dweight++;
            if (dl_dout == dl_dout_end) {
                dl_dout = dl_dout_reset;
                in++;
                if (dl_dweight == dl_dweight_end) {
                    break;
                }
            }
        }
    }
}


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
    size_t grads_offset = offset;
    offset += align(sizeof(struct grads));
    size_t activations_back_offset = offset;
    offset += align(sizeof(struct activations_back));
#undef align
    char* raw_mem = malloc(offset);
    struct decoder* decoder = (struct decoder*)(raw_mem + decoder_offset);
    struct data* data = (struct data*)(raw_mem + data_offset);
    char* json_raw = (char*)(raw_mem + json_raw_offset);
    float* params_raw = (float*)(raw_mem + params_raw_offset);
    struct params* params = (struct params*)(raw_mem + params_offset);
    struct activations* activations = (struct activations*)(raw_mem + activations_offset);
    struct grads* grads = (struct grads*)(raw_mem + grads_offset);
    struct activations_back* activations_back = (struct activations_back*)(raw_mem + activations_back_offset);

    

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
            
            sprintf(temp_name, "h.%u.ln_1.weight", layer_i);
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
    uint16_t* expected = data->tokens + 1;

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
    // double sum = 0.0;
    // for (size_t i = 0; i < inp_size * d_model; i++) {
    //     sum += (double)((float*)activations->embedding.out)[i];
    // }
    // printf("sum: %f\n", sum);

   
    for (int layer_idx = 0; layer_idx < n_layers; layer_idx++) {

        float* layer_in = layer_idx == 0 ? (float*)activations->embedding.out : (float*)activations->h[layer_idx - 1].res_2.out;

        // here insert layer norm 1

        layer_norm(&(struct ln){
            .in = layer_in,
            .weight = params->h[layer_idx].ln_1.weight,
            .bias = params->h[layer_idx].ln_1.bias,
            .out = (float*)activations->h[layer_idx].ln_1.out,
            .inv_std = (float*)activations->h[layer_idx].ln_1.inv_std,
            .mean = (float*)activations->h[layer_idx].ln_1.mean,
            .in_size = d_model,
            .sample_size = inp_size
        });

        // linear transformation

        linear(&(struct linear){
            .in = (float*)activations->h[layer_idx].ln_1.out,
            .weights = params->h[layer_idx].attn.c_attn.weight,
            .bias = params->h[layer_idx].attn.c_attn.bias,
            .out = (float*)activations->h[layer_idx].attn.c_attn.out,
            .in_size = d_model,
            .out_size = 3 * d_model,
            .sample_size = inp_size
        });

        // multihead attention

        memset(activations->h[layer_idx].attn.z.out, 0, sizeof(activations->h[layer_idx].attn.z.out[0]));

        for (size_t head_i = 0; head_i < 12; head_i++) {
            for (size_t q_i = 0; q_i < inp_size; q_i++) {

                float softmax_max = -INFINITY;
                
                for (size_t k_i = 0; k_i <= q_i; k_i++) {
                    // first block of the matrix are the q vectors point to d_model
                    float* q = (float*)activations->h[layer_idx].attn.c_attn.out + q_i * 3 * d_model + head_i * d_k;
                    float* q_end = q + d_k;
                    // second block of the matrix are the k vectors point to d_model
                    float* k = (float*)activations->h[layer_idx].attn.c_attn.out + k_i * 3 * d_model + d_model + head_i * d_k; // acess block of k vectors for each head
                    
                    
                    float dot  = 0.0f;
                    for (; q != q_end; q++, k++) {
                        dot += *q * *k;  
                    }
                    
                    dot *= inv_sqrt_k;
                    activations->h[layer_idx].attn.attn.out[head_i][q_i][k_i] = dot;

                    if (dot > softmax_max) {
                        softmax_max = dot;
                    }
                }
                
                float softmax_sum = 0.0f;

                for (size_t k_i = 0; k_i <= q_i; k_i++){
                    float e = activations->h[layer_idx].attn.attn.out[head_i][q_i][k_i];

                    float softmax_exp_i = expf(e - softmax_max);
                    activations->h[layer_idx].attn.s.out[head_i][q_i][k_i] = softmax_exp_i;

                    softmax_sum += softmax_exp_i;

                }

                float inv_softmax_sum = 1.0f / softmax_sum;

                for (size_t k_i = 0; k_i <= q_i; k_i++) {
                    activations->h[layer_idx].attn.s.out[head_i][q_i][k_i] *= inv_softmax_sum;
                }

                for (size_t v_i = 0; v_i <= q_i; v_i++) {
                    // last block of the matrix are the v vectors point to 2 * d_model
                    float* v = (float*)activations->h[layer_idx].attn.c_attn.out + v_i * 3 * d_model + 2 * d_model + head_i * d_k; // access block of v vectors for each head
                    float* z = (float*)activations->h[layer_idx].attn.z.out + q_i * d_model + head_i * d_k;
                    float* v_end = v + d_k;
                    
                    float fctr = activations->h[layer_idx].attn.s.out[head_i][q_i][v_i];

                    for (; v != v_end; v++, z++) {
                        *z += *v * fctr;
                    }
                }
            }      
        }
        
        linear(&(struct linear){
            .in = (float*)activations->h[layer_idx].attn.z.out,
            .weights = params->h[layer_idx].attn.c_proj.weight,
            .bias = params->h[layer_idx].attn.c_proj.bias,
            .out = (float*)activations->h[layer_idx].attn.c_proj.out,
            .in_size = d_model,
            .out_size = d_model,
            .sample_size = inp_size
        });

        // residual connection
        const float* in_1 = layer_in;
        const float* in_2 = (float*)activations->h[layer_idx].attn.c_proj.out;

        float* out = (float*)activations->h[layer_idx].res_1.out;

        float* out_end = out + inp_size * d_model;

        for (; out != out_end; out++, in_1++, in_2++) {
            *out = *in_1 + *in_2;
        }

        layer_norm(&(struct ln){
            .in = (float*)activations->h[layer_idx].res_1.out,
            .weight = params->h[layer_idx].ln_2.weight,
            .bias = params->h[layer_idx].ln_2.bias,
            .out = (float*)activations->h[layer_idx].ln_2.out,
            .mean = (float*)activations->h[layer_idx].ln_2.mean,
            .inv_std = (float*)activations->h[layer_idx].ln_2.inv_std,
            .in_size = d_model,
            .sample_size = inp_size
        });

        // mlp c_fc linear transformation

        linear(&(struct linear){
            .in = (float*)activations->h[layer_idx].ln_2.out,
            .weights = params->h[layer_idx].mlp.c_fc.weight,
            .bias = params->h[layer_idx].mlp.c_fc.bias,
            .out = (float*)activations->h[layer_idx].mlp.c_fc.out,
            .in_size = d_model,
            .out_size = 4 * d_model,
            .sample_size = inp_size
        });

        // gelu activation

        for (size_t i = 0; i < inp_size; i++) {
            float* in = (float*)activations->h[layer_idx].mlp.c_fc.out + i * 4 * d_model; // access the rows of the matrix
            float* out = (float*)activations->h[layer_idx].mlp.gelu.out + i * 4 * d_model;
            float* out_end = out + inp_size * 4 * d_model;

            for (; out != out_end; out++, in++) {
                float cdf = 0.5 * (1.0 + erff(*in * (float)M_SQRT1_2)); // M_SQRT1_2 def in math.h
                *out = *in * cdf;
            }
        }
    

    // mlp c_proj linear transformation

        linear(&(struct linear){
            .in = (float*)activations->h[layer_idx].mlp.gelu.out,
            .weights = params->h[layer_idx].mlp.c_proj.weight,
            .bias = params->h[layer_idx].mlp.c_proj.bias,
            .out = (float*)activations->h[layer_idx].mlp.c_proj.out,
            .in_size = 4 * d_model,
            .out_size = d_model,
            .sample_size = inp_size
        });

        {
            const float* in_1 = (float*)activations->h[layer_idx].res_1.out;
            const float* in_2 = (float*)activations->h[layer_idx].mlp.c_proj.out;

            float* out = (float*)activations->h[layer_idx].res_2.out;

            float* out_end = out + inp_size * d_model;

            for (; out != out_end; out++, in_1++, in_2++) {
                *out = *in_1 + *in_2;
            }
        }
    }

    layer_norm(&(struct ln){
        .in = (float*)activations->h[11].res_2.out,
        .weight = params->ln_f.weight,
        .bias = params->ln_f.bias,
        .out = (float*)activations->ln_f.out,
        .mean = activations->ln_f.mean,
        .inv_std = activations->ln_f.inv_std,
        .in_size = d_model,
        .sample_size = inp_size
    });

    // unembedding

    for (size_t i = 0; i < inp_size; i++) {
        const float* in = (float*)activations->ln_f.out + i * d_model;
        const float* in_end = in + d_model;
        const float* in_reset = in;

        const float* weight = params->wte.weight;
        const float* weight_end = weight + d_vocab * d_model;

        float* out = (float*)activations->unembedding.out + i * d_vocab;

        float dot = 0.0f;

        while (true){
            dot += *in * *weight;
            in++;
            weight++;
            if (in == in_end) {
                in = in_reset;
                *out = dot;
                dot = 0.0f;
                out++;
                if (weight == weight_end) {
                    break;
                }
            }
        }
    }

    for (size_t i = 0; i < inp_size; i++) {
        const float* in = (float*)activations->unembedding.out + i * d_vocab;
        const float* in_end = in + d_vocab;
        const float* in_reset = in;

        float softmax_max = -INFINITY;
        for (; in != in_end; in++) {
            if (*in > softmax_max) {
                softmax_max = *in;
            }
        }

        float softmax_sum = 0.0f;

        float* out = (float*)activations->prob.out + i * d_vocab;
        float* out_end = out + d_vocab;
        float* out_reset = out;

        for (in = in_reset; in != in_end; in++, out++) {
            float e = expf(*in - softmax_max);
            *out = e;
            softmax_sum += e;
        }
        float inv_softmax_sum = 1.0f / softmax_sum;
        
        for (out = out_reset; out != out_end; out++) {
            *out *= inv_softmax_sum;
        }

    }

    float loss = 0.0f;
    for (size_t i = 0; i < inp_size; i++) {
        uint16_t token = expected[i];
        float ce = -logf(activations->prob.out[i][token]);
        loss += ce; 
    }

    // avergae loss
    loss /= (float)inp_size;
    printf("loss: %f\n", loss);

    memset(activations_back, 0, sizeof(struct activations_back));
    memset(grads, 0, sizeof(struct grads));

    // backwards pass for cross entropy loss

    memcpy(activations_back->unembedding.out, activations->prob.out, inp_size * d_vocab * sizeof(float));

    for (size_t i = 0; i < inp_size; i++) {
        uint16_t token = expected[i];
        activations_back->unembedding.out[i][token] -= 1.0f;
    }

    // backwards pass for unembedding


    float inv_inp_size = 1.0f / (float)inp_size;
    
    // gradients with respect to the input
    for (size_t i = 0; i < inp_size; i++) {
        const float* in = (float*)activations->ln_f.out + i * d_model;
        float* dl_din = (float*)activations_back->ln_f.out + i * d_model;
        float* dl_din_end = dl_din + d_model;
        float* dl_din_reset = dl_din;

        const float* weight = params->wte.weight;
        const float* weight_end = weight + d_vocab * d_model;
        float* dl_dweight = (float*)grads->wte.weight;

        const float* dl_dout = (float*)activations_back->unembedding.out + i * d_vocab;
        
        while (true){
            *dl_din += *dl_dout * *weight;
            *dl_dweight += *dl_dout * *in * inv_inp_size;
            in++;
            dl_din++;
            weight++;
            dl_dweight++;
            if (dl_din == dl_din_end) {
                dl_din = dl_din_reset;
                dl_dout++;
                if (weight == weight_end) {
                    break;
                }
            }
        }
        for (dl_din = dl_din_reset; dl_din != dl_din_end; dl_din++) {
            *dl_din *= 1.0f * inv_inp_size;
        }
    }

    // backwards pass for layer norm f
    // for (size_t i = 0; i < inp_size; i++) {
    //     float* dl_din = (float*)activations_back->h[11].res_2.out + i * d_model;
    //     const float* in = (float*)activations->h[11].res_2.out + i * d_model;
    //     const float* in_reset = in;
        

    //     float* dl_dout = (float*)activations_back->ln_f.out + i * d_model;
    //     float* dl_dout_end = dl_dout + d_model;
    //     float* dl_dout_reset = dl_dout;

    //     float* dl_dbias = (float*)grads->ln_f.bias;
    //     float* dl_dweight = (float*)grads->ln_f.weight;
    //     const float* weight = params->ln_f.weight;
    //     const float* weight_reset = weight;

    //     float mean = activations->ln_f.mean[i];
    //     float inv_std = activations->ln_f.inv_std[i];

    //     float mean_partial = 0.0f;
    //     float std_partial = 0.0f;

    //     for (; dl_dout != dl_dout_end; dl_dout++, dl_dbias++, dl_dweight++, weight++, in++) {
    //         float dl_din_norm = *dl_dout * *weight; // dL/din_norm
    //         float in_norm = (*in - mean) * inv_std; // in_norm
    //         *dl_dbias += *dl_dout;
    //         *dl_dweight += *dl_dout * in_norm;

            
    //         mean_partial += dl_din_norm;
    //         std_partial += dl_din_norm * in_norm;

    //     }

    //     mean_partial /= (float)d_model;
    //     std_partial /= (float)d_model;

    //     dl_dout = dl_dout_reset;
    //     in = in_reset;
    //     weight = weight_reset;

    //     for (; dl_dout != dl_dout_end; dl_dout++, in++, dl_din++, weight++) {
    //         float dl_din_norm = *dl_dout * *weight;
    //         float in_norm = (*in - mean) * inv_std;

    //         float dl_din_ = 0.0f;
    //         dl_din_ += dl_din_norm;
    //         dl_din_ -= mean_partial;
    //         dl_din_ -= std_partial * in_norm;
    //         dl_din_ *= inv_std;
    //         *dl_din = dl_din_;


    //     }

    // }

    layer_norm_back(&(struct ln_back){
        .in = (float*)activations->h[11].res_2.out,
        .weight = params->ln_f.weight,
        .bias = params->ln_f.bias,
        .dl_dout = (float*)activations_back->ln_f.out,
        .inv_std = activations->ln_f.inv_std,
        .mean = activations->ln_f.mean,
        .dl_din = (float*)activations_back->h[11].res_2.out,
        .dl_dweight = (float*)grads->ln_f.weight,
        .dl_dbias = (float*)grads->ln_f.bias,
        .in_size = d_model,
        .sample_size = inp_size
    });


    // backwards pass for the 12 layers

    // for (size_t i = 0; i < inp_size; i++){
    //     float* dl_dweight = (float*)grads->h[11].mlp.c_proj.weight;
    //     float* dl_dweight_end = dl_dweight + 4 * d_model * d_model;
    //     float* dl_dbias = (float*)grads->h[11].mlp.c_proj.bias;
    //     const float* dl_dout = (float*)activations_back->h[11].res_2.out + i * d_model;
    //     const float* dl_dout_end = dl_dout + d_model;
    //     const float* dl_dout_reset = dl_dout;

    //     const float* in = (float*)activations->h[11].mlp.gelu.out + i * 4 * d_model;
    //     float* dl_din = (float*)activations_back->h[11].mlp.gelu.out + i * 4 * d_model;

    //     for (; dl_dout != dl_dout_end; dl_dout++, dl_dbias++) { //  dl_dweight++,
    //         *dl_dbias += *dl_dout;
    //     }

    //     dl_dout = dl_dout_reset;

    //     while (true) {
    //         *dl_dweight += *in * *dl_dout;
    //         dl_dout++;
    //         dl_dweight++;
    //         if (dl_dout == dl_dout_end) {
    //             dl_dout = dl_dout_reset;
    //             in++;
    //             if (dl_dweight == dl_dweight_end) {
    //                 break;
    //             }
    //         }
    //     }
    // }




    double sum = 0.0;
    for (size_t i = 0; i < 4 * d_model * d_model; i++) {
        sum += (double)fabsf(((float*)grads->h[11].mlp.c_proj.weight)[i]);
    }  

    printf("sum: %.16f\n", sum);



    // float max_v = -INFINITY;
    // size_t max_idx = 0;
    // float* row = (float*)activations->unembedding.out + (inp_size - 1) * d_vocab;

    // // argmax
    // for (size_t s = 0; s < d_vocab; s++) {
    //     if (row[s] > max_v) {
    //         max_v = row[s];
    //         max_idx = s;
    //     }
    // }

    // for (size_t i = 0; i < inp_size; i++) {
    //     uint16_t token = inp[i];
    //     printf("%.*s", decoder->tokens[token].length, decoder->raw + decoder->tokens[token].offset);
    // }
    // printf("\n");
    // printf("%.*s\n", decoder->tokens[max_idx].length, decoder->raw + decoder->tokens[max_idx].offset);

    // printf("max_idx: %zu\n", max_idx);

    // cross entropy loss one hot vector for target ce(x) = -log(softmax(x)[target])

    // step 1 compute softmax



    // double sum = 0.0;
    // for (size_t i = 0; i < inp_size * d_vocab; i++) {
    //     sum += (double)((float*)activations->unembedding.out)[i];
    // }  

    // printf("sum: %f\n", sum);
    return 0;
}

