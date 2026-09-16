#pragma once

#include "ggml.h"

#include <fstream>
#include <vector>
#include <string>

enum ggml_ftype ggml_parse_ftype(const char * str);

void ggml_print_ftypes(FILE * fp = stderr);

// Fit this model's own lattice levels before quantizing, and bind them. False if ftype is not a
// lattice type or the file has no weights to fit on. Leaves finp where it found it.
bool ggml_common_fit_levels(
        std::ifstream & finp,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip,
        std::vector<float> & levels);

bool ggml_common_quantize_0(
        std::ifstream & finp,
        std::ofstream & fout,
        const ggml_ftype ftype,
        const std::vector<std::string> & to_quant,
        const std::vector<std::string> & to_skip);
