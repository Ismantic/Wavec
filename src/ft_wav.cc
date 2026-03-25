#include "ft_wav.h"
#include <iostream>
#include <cstring>
#include <cstdlib>

void PrintUsage() {
    std::cerr << "Usage: wavec [options] <input> <output>\n"
              << "Options:\n"
              << "  -dim <int>      Vector dimension (default: 100)\n"
              << "  -window <int>   Context window size (default: 5)\n"
              << "  -mincount <int> Minimum word frequency (default: 5)\n"
              << "  -threads <int>  Number of threads (default: 4)\n"
              << "  -iter <int>     Training iterations (default: 5)\n"
              << "  -sample <float> Subsampling threshold (default: 1e-3)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        PrintUsage();
        return 1;
    }

    int dim = 100;
    int window = 5;
    int mincount = 5;
    int threads = 4;
    int iter = 5;
    float sample = 1e-3;

    int i = 1;
    while (i < argc - 2) {
        if (std::strcmp(argv[i], "-dim") == 0 && i + 1 < argc - 2) {
            dim = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-window") == 0 && i + 1 < argc - 2) {
            window = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-mincount") == 0 && i + 1 < argc - 2) {
            mincount = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-threads") == 0 && i + 1 < argc - 2) {
            threads = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-iter") == 0 && i + 1 < argc - 2) {
            iter = std::atoi(argv[++i]);
        } else if (std::strcmp(argv[i], "-sample") == 0 && i + 1 < argc - 2) {
            sample = std::atof(argv[++i]);
        } else {
            std::cerr << "Unknown option: " << argv[i] << "\n";
            PrintUsage();
            return 1;
        }
        ++i;
    }

    std::string input = argv[argc - 2];
    std::string output = argv[argc - 1];

    std::cerr << "Input:    " << input << "\n"
              << "Output:   " << output << "\n"
              << "dim=" << dim << " window=" << window
              << " mincount=" << mincount << " threads=" << threads
              << " iter=" << iter << " sample=" << sample << "\n";

    wavec::FastText model;
    model.SetVecSize(dim);
    model.SetWindow(window);
    model.SetMinCount(mincount);

    model.SetCores(threads);
    model.SetIter(iter);
    model.SetSample(sample);

    model.Fit(input, output);
    return 0;
}
