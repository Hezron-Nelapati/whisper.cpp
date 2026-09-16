#include "common-ggml.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <map>
#include <regex>

static const std::map<std::string, enum ggml_ftype> GGML_FTYPE_MAP = {
    {"q4_0", GGML_FTYPE_MOSTLY_Q4_0},
    {"q4_1", GGML_FTYPE_MOSTLY_Q4_1},
    {"q5_0", GGML_FTYPE_MOSTLY_Q5_0},
    {"q5_1", GGML_FTYPE_MOSTLY_Q5_1},
    {"q8_0", GGML_FTYPE_MOSTLY_Q8_0},
    {"q2_k", GGML_FTYPE_MOSTLY_Q2_K},
    {"q3_k", GGML_FTYPE_MOSTLY_Q3_K},
    {"q4_k", GGML_FTYPE_MOSTLY_Q4_K},
    {"q5_k", GGML_FTYPE_MOSTLY_Q5_K},
    {"q6_k", GGML_FTYPE_MOSTLY_Q6_K},
    {"neuron_v4", GGML_FTYPE_MOSTLY_NEURON_V4},
    {"neuron_l4", GGML_FTYPE_MOSTLY_NEURON_L4},
    {"neuron_l5", GGML_FTYPE_MOSTLY_NEURON_L5},
    {"neuron_l6", GGML_FTYPE_MOSTLY_NEURON_L6},
    {"neuron_l7", GGML_FTYPE_MOSTLY_NEURON_L7},
};

void ggml_print_ftypes(FILE * fp) {
    for (auto it = GGML_FTYPE_MAP.begin(); it != GGML_FTYPE_MAP.end(); it++) {
        fprintf(fp, "  type = \"%s\" or %d\n", it->first.c_str(), it->second);
    }
}

enum ggml_ftype ggml_parse_ftype(const char * str) {
    enum ggml_ftype ftype;
    // Names first, numbers only if the string is one. Keying on a leading 'q' sent every
    // non-q name through atoi, so "neuron_v4" silently became 0 (all f32).
    const auto it = GGML_FTYPE_MAP.find(str);
    if (it != GGML_FTYPE_MAP.end()) {
        ftype = it->second;
    } else if (str[0] >= '0' && str[0] <= '9') {
        ftype = (enum ggml_ftype) atoi(str);
    } else {
        fprintf(stderr, "%s: unknown ftype '%s'\n", __func__, str);
        return GGML_FTYPE_UNKNOWN;
    }

    return ftype;
}

// Fit the lattice levels on this model's own weights. Lloyd with the real encoder in the loop:
// encode sampled blocks with the current levels, then move each magnitude to the scale-weighted
// mean of the values that picked it. The recipe is llama-quantize's: 12 rounds from the uniform
// grid at |x/g|^-2.5, top level scaled to 1. Leaves finp where it found it.
bool ggml_common_fit_levels(
        std::ifstream & finp,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip,
        std::vector<float> & levels) {
    const ggml_type T = ggml_ftype_to_ggml_type(ftype);
    const int       NL = ggml_neuron_l_n_levels(T);
    if (NL <= 0) {
        return false;
    }
    const int    BLK    = (int) ggml_blck_size(T);
    const size_t TSZ    = ggml_type_size(T);
    const int    H      = NL / 2;
    const size_t NBLK   = 32768;
    // The recipe was measured at 16 levels. The weight floor is what the objective calls "small",
    // so it is a property of the model, not of how many levels the codec has: tying it to H made
    // every type fit a different objective. Overridable while that is being measured.
    const int    ROUNDS = getenv("NEURON_FIT_ROUNDS") ? atoi(getenv("NEURON_FIT_ROUNDS")) : 12;
    const float  ALPHA  = getenv("NEURON_FIT_ALPHA")  ? (float) atof(getenv("NEURON_FIT_ALPHA"))  : 2.5f;
    const float  FLOOR  = getenv("NEURON_FIT_FLOOR")  ? (float) atof(getenv("NEURON_FIT_FLOOR"))  : 1.0f / 16.0f;
    if (getenv("NEURON_FIT_DISABLE")) {
        return false;                        // keep the shipped table, for A/B against the fit
    }

    const std::streampos start = finp.tellg();

    // one pass for where the weights are, so the sample can be spread over every layer
    struct src_tensor { std::streampos off; int64_t nblk; int32_t ttype; };
    std::vector<src_tensor> src;
    while (true) {
        int32_t n_dims, length, ttype;
        finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char *>(&length), sizeof(length));
        finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        if (finp.eof()) {
            break;
        }
        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }
        std::string name(length, 0);
        finp.read(&name[0], length);

        bool quantize = false;
        for (const auto & r : to_quant) {
            if (std::regex_match(name, std::regex(r))) { quantize = true; break; }
        }
        for (const auto & r : to_skip) {
            if (std::regex_match(name, std::regex(r))) { quantize = false; break; }
        }
        quantize &= (n_dims == 2) && (ttype == GGML_TYPE_F32 || ttype == GGML_TYPE_F16) && (ne[0] % BLK == 0);

        const size_t bpe = (ttype == GGML_TYPE_F32) ? sizeof(float) : sizeof(uint16_t);
        if (quantize) {
            src.push_back({ finp.tellg(), nelements / BLK, ttype });
        }
        finp.seekg((size_t) nelements * bpe, std::ios::cur);
    }
    finp.clear();

    if (src.empty()) {
        finp.seekg(start);
        return false;
    }

    // read only the sampled blocks, spread evenly through each tensor
    const size_t per = std::max<size_t>(1, NBLK / src.size());
    std::vector<float>       X;
    std::vector<ggml_fp16_t> half(BLK);
    X.reserve(NBLK * BLK);
    for (const auto & t : src) {
        const size_t take = std::min<size_t>(per, (size_t) t.nblk);
        const size_t bpe  = (t.ttype == GGML_TYPE_F32) ? sizeof(float) : sizeof(uint16_t);
        for (size_t s = 0; s < take; ++s) {
            const size_t b = (size_t) ((double) s * t.nblk / take);
            finp.seekg(t.off + (std::streamoff) (b * BLK * bpe));
            X.resize(X.size() + BLK);
            float * dst = X.data() + X.size() - BLK;
            if (t.ttype == GGML_TYPE_F32) {
                finp.read(reinterpret_cast<char *>(dst), BLK * sizeof(float));
            } else {
                finp.read(reinterpret_cast<char *>(half.data()), BLK * sizeof(ggml_fp16_t));
                for (int i = 0; i < BLK; ++i) {
                    dst[i] = ggml_fp16_to_fp32(half[i]);
                }
            }
        }
    }
    finp.clear();
    finp.seekg(start);

    const size_t nb = X.size() / BLK;
    if (nb == 0) {
        return false;
    }

    std::vector<float> L(2 * H);
    for (int j = 0; j < H; ++j) {
        L[H + j]     =  (2.0f * j + 1.0f) / (2.0f * H - 1.0f);
        L[H - 1 - j] = -L[H + j];
    }
    std::vector<uint8_t> q(nb * TSZ);
    std::vector<float>   R(nb * BLK);
    std::vector<uint8_t> nidx(nb * BLK);
    ggml_quantize_init(T);

    for (int r = 0; r < ROUNDS; ++r) {
        ggml_neuron_l_set_levels(T, L.data());
        ggml_quantize_chunk(T, X.data(), q.data(), 0, (int64_t) nb, BLK, nullptr);
        ggml_get_type_traits(T)->to_float(q.data(), R.data(), (int64_t) nb * BLK);
        ggml_neuron_l_get_indices(T, q.data(), (int64_t) nb * BLK, nidx.data());

        // reconstruction is g * L[n] and no level is zero, so g = rec / L[n]
        const float tfloor = FLOOR;
        std::vector<double> num(H, 0.0), den(H, 0.0);
        double sse = 0.0, tot = 0.0;
        for (size_t i = 0; i < nb * (size_t) BLK; ++i) {
            const float x = X[i];
            const float y = R[i];
            const int   n = nidx[i];
            const int   j = n >= H ? n - H : H - 1 - n;
            const float g = y / L[n];
            const double e = (double) x - y;
            sse += e * e;
            tot += (double) x * x;
            if (!(g > 0.0f)) {
                continue;
            }
            const float w = std::pow(std::max(std::fabs(x) / g, tfloor), -ALPHA);
            num[j] += (double) w * g * std::fabs(x);
            den[j] += (double) w * g * g;
        }
        std::vector<float> mag(H);
        for (int j = 0; j < H; ++j) {
            mag[j] = den[j] > 0.0 ? (float) (num[j] / den[j]) : L[H + j];
        }
        std::sort(mag.begin(), mag.end());
        for (int j = 0; j < H; ++j) {
            L[H + j]     =  mag[j];
            L[H - 1 - j] = -mag[j];
        }
        fprintf(stderr, "%s: round %2d  rel mse %.6e\n", __func__, r + 1, sse / std::max(tot, 1e-30));
    }
    // d and the levels share one scale, so state the table with its top at 1
    const float top = L[2 * H - 1] > 0.0f ? L[2 * H - 1] : 1.0f;
    for (float & v : L) {
        v /= top;
    }
    fprintf(stderr, "%s: %s levels fitted on %zu blocks of this model\n", __func__, ggml_type_name(T), nb);
    levels = L;
    ggml_neuron_l_set_levels(T, levels.data());
    return true;
}

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip) {

    ggml_type qtype = GGML_TYPE_F32;

    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: qtype = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: qtype = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q5_0: qtype = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: qtype = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: qtype = GGML_TYPE_Q8_0; break;
        case GGML_FTYPE_MOSTLY_Q2_K: qtype = GGML_TYPE_Q2_K; break;
        case GGML_FTYPE_MOSTLY_Q3_K: qtype = GGML_TYPE_Q3_K; break;
        case GGML_FTYPE_MOSTLY_Q4_K: qtype = GGML_TYPE_Q4_K; break;
        case GGML_FTYPE_MOSTLY_Q5_K: qtype = GGML_TYPE_Q5_K; break;
        case GGML_FTYPE_MOSTLY_Q6_K: qtype = GGML_TYPE_Q6_K; break;
        case GGML_FTYPE_MOSTLY_NEURON_V4: qtype = GGML_TYPE_NEURON_V4; break;
        case GGML_FTYPE_MOSTLY_NEURON_L4: qtype = GGML_TYPE_NEURON_L4; break;
        case GGML_FTYPE_MOSTLY_NEURON_L5: qtype = GGML_TYPE_NEURON_L5; break;
        case GGML_FTYPE_MOSTLY_NEURON_L6: qtype = GGML_TYPE_NEURON_L6; break;
        case GGML_FTYPE_MOSTLY_NEURON_L7: qtype = GGML_TYPE_NEURON_L7; break;
        case GGML_FTYPE_UNKNOWN:
        case GGML_FTYPE_ALL_F32:
        case GGML_FTYPE_MOSTLY_F16:
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
        case GGML_FTYPE_MOSTLY_IQ2_XXS:
        case GGML_FTYPE_MOSTLY_IQ2_XS:
        case GGML_FTYPE_MOSTLY_IQ2_S:
        case GGML_FTYPE_MOSTLY_IQ3_XXS:
        case GGML_FTYPE_MOSTLY_IQ3_S:
        case GGML_FTYPE_MOSTLY_IQ1_S:
        case GGML_FTYPE_MOSTLY_IQ4_NL:
        case GGML_FTYPE_MOSTLY_IQ4_XS:
        case GGML_FTYPE_MOSTLY_IQ1_M:
        case GGML_FTYPE_MOSTLY_BF16:
        case GGML_FTYPE_MOSTLY_MXFP4:
        case GGML_FTYPE_MOSTLY_NVFP4:
        case GGML_FTYPE_MOSTLY_Q1_0:
        case GGML_FTYPE_MOSTLY_Q2_0:
                {
                    fprintf(stderr, "%s: invalid model type %d\n", __func__, ftype);
                    return false;
                }
    };

    if (!ggml_is_quantized(qtype)) {
        fprintf(stderr, "%s: invalid quantization type %d (%s)\n", __func__, qtype, ggml_type_name(qtype));
        return false;
    }

    size_t total_size_org = 0;
    size_t total_size_new = 0;

    std::vector<float> work;

    std::vector<uint8_t>     data_u8;
    std::vector<ggml_fp16_t> data_f16;
    std::vector<float>       data_f32;

    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ttype;

        finp.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        finp.read(reinterpret_cast<char *>(&length), sizeof(length));
        finp.read(reinterpret_cast<char *>(&ttype),  sizeof(ttype));

        if (finp.eof()) {
            break;
        }

        int32_t nelements = 1;
        int32_t ne[4] = { 1, 1, 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            finp.read (reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
            nelements *= ne[i];
        }

        std::string name(length, 0);
        finp.read (&name[0], length);

        printf("%64s - [%5d, %5d, %5d], type = %6s ", name.data(), ne[0], ne[1], ne[2], ggml_type_name((ggml_type) ttype));

        bool quantize = false;

        // check if we should quantize this tensor
        for (const auto & s : to_quant) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = true;
                break;
            }
        }

        // check if we should skip this tensor
        for (const auto & s : to_skip) {
            if (std::regex_match(name, std::regex(s))) {
                quantize = false;
                break;
            }
        }

        // quantize only 2D tensors
        quantize &= (n_dims == 2);

        if (quantize) {
            if (ttype != GGML_TYPE_F32 && ttype != GGML_TYPE_F16) {
                fprintf(stderr, "%s: unsupported ttype %d (%s) for integer quantization\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                return false;
            }

            if (ttype == GGML_TYPE_F16) {
                data_f16.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f16.data()), nelements * sizeof(ggml_fp16_t));
                data_f32.resize(nelements);
                for (int i = 0; i < nelements; ++i) {
                    data_f32[i] = ggml_fp16_to_fp32(data_f16[i]);
                }
            } else {
                data_f32.resize(nelements);
                finp.read(reinterpret_cast<char *>(data_f32.data()), nelements * sizeof(float));
            }

            ttype = qtype;
        } else {
            const int bpe = (ttype == 0) ? sizeof(float) : sizeof(uint16_t);

            data_u8.resize(nelements*bpe);
            finp.read(reinterpret_cast<char *>(data_u8.data()), nelements * bpe);
        }

        fout.write(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fout.write(reinterpret_cast<char *>(&length), sizeof(length));
        fout.write(reinterpret_cast<char *>(&ttype),  sizeof(ttype));
        for (int i = 0; i < n_dims; ++i) {
            fout.write(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }
        fout.write(&name[0], length);

        if (quantize) {
            work.resize(nelements); // for quantization

            size_t cur_size = 0;
            switch ((ggml_type) ttype) {
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                case GGML_TYPE_Q2_K:
                case GGML_TYPE_Q3_K:
                case GGML_TYPE_Q4_K:
                case GGML_TYPE_Q5_K:
                case GGML_TYPE_Q6_K:
                case GGML_TYPE_NEURON_V4:
                case GGML_TYPE_NEURON_L4:
                case GGML_TYPE_NEURON_L5:
                case GGML_TYPE_NEURON_L6:
                case GGML_TYPE_NEURON_L7:
                    {
                        cur_size = ggml_quantize_chunk((ggml_type) ttype, data_f32.data(), work.data(), 0, nelements/ne[0], ne[0], nullptr);
                    } break;
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_I8:
                case GGML_TYPE_I16:
                case GGML_TYPE_I32:
                case GGML_TYPE_I64:
                case GGML_TYPE_F64:
                case GGML_TYPE_Q8_1:
                case GGML_TYPE_Q8_K:
                case GGML_TYPE_IQ2_XXS:
                case GGML_TYPE_IQ2_XS:
                case GGML_TYPE_IQ2_S:
                case GGML_TYPE_IQ3_XXS:
                case GGML_TYPE_IQ3_S:
                case GGML_TYPE_IQ1_S:
                case GGML_TYPE_IQ4_NL:
                case GGML_TYPE_IQ4_XS:
                case GGML_TYPE_IQ1_M:
                case GGML_TYPE_BF16:
                case GGML_TYPE_TQ1_0:
                case GGML_TYPE_TQ2_0:
                case GGML_TYPE_MXFP4:
                case GGML_TYPE_NVFP4:
                case GGML_TYPE_Q1_0:
                case GGML_TYPE_Q2_0:
                case GGML_TYPE_COUNT:
                    {
                        fprintf(stderr, "%s: unsupported quantization type %d (%s)\n", __func__, ttype, ggml_type_name((ggml_type) ttype));
                        return false;
                    }
            }

            fout.write(reinterpret_cast<char *>(work.data()), cur_size);
            total_size_new += cur_size;

            printf("size = %8.2f MB -> %8.2f MB\n", nelements * sizeof(float)/1024.0/1024.0, cur_size/1024.0/1024.0);
        } else {
            printf("size = %8.3f MB\n", data_u8.size()/1024.0/1024.0);
            fout.write(reinterpret_cast<char *>(data_u8.data()), data_u8.size());
            total_size_new += data_u8.size();
        }

        total_size_org += nelements * sizeof(float);
    }

    printf("%s: model size  = %8.2f MB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MB | ftype = %d (%s)\n", __func__, total_size_new/1024.0/1024.0, ftype, ggml_type_name(qtype));

    return true;
}
