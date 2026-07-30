#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <cmath>

struct WordVec {
    std::string word;
    std::vector<float> vec;
};

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr << "Usage: sim <model.vec> [topk]\n";
        return 1;
    }

    std::string model_file = argv[1];
    int topk = argc >= 3 ? std::atoi(argv[2]) : 10;
    if (topk <= 0) {
        std::cerr << "Error: topk must be positive.\n";
        return 1;
    }

    // Load vectors
    std::ifstream fin(model_file);
    if (!fin) {
        std::cerr << "Error: Cannot open model file: " << model_file << "\n";
        return 1;
    }
    int vocab_size = 0, dim = 0;
    if (!(fin >> vocab_size >> dim) || vocab_size <= 0 || dim <= 0) {
        std::cerr << "Error: Invalid model header.\n";
        return 1;
    }

    std::vector<WordVec> words(vocab_size);
    std::unordered_map<std::string, int> word2id;

    std::cerr << "Loading " << vocab_size << " words, dim=" << dim << "...\n";

    for (int i = 0; i < vocab_size; i++) {
        fin >> words[i].word;
        words[i].vec.resize(dim);
        for (int j = 0; j < dim; j++) {
            fin >> words[i].vec[j];
        }
        if (!fin || words[i].word.empty()) {
            std::cerr << "Error: Invalid or truncated model data.\n";
            return 1;
        }
        // L2 normalize
        float norm = 0;
        for (int j = 0; j < dim; j++) norm += words[i].vec[j] * words[i].vec[j];
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (int j = 0; j < dim; j++) words[i].vec[j] /= norm;
        }
        word2id[words[i].word] = i;
    }

    std::cerr << "Ready. Enter a word (or quit):\n";

    // REPL
    std::string query;
    while (std::cout << "> " && std::getline(std::cin, query)) {
        if (query.empty() || query == "quit" || query == "exit") break;

        auto it = word2id.find(query);
        if (it == word2id.end()) {
            std::cout << "Not in vocabulary.\n";
            continue;
        }

        int qid = it->second;
        const auto& qvec = words[qid].vec;

        // Compute cosine similarity with all words
        std::vector<std::pair<float, int>> scores;
        scores.reserve(vocab_size);
        for (int i = 0; i < vocab_size; i++) {
            if (i == qid) continue;
            float dot = 0;
            for (int j = 0; j < dim; j++) dot += qvec[j] * words[i].vec[j];
            scores.emplace_back(dot, i);
        }

        int count = std::min(topk, static_cast<int>(scores.size()));
        std::partial_sort(scores.begin(), scores.begin() + count, scores.end(),
                          [](const auto& a, const auto& b) { return a.first > b.first; });

        for (int i = 0; i < count; i++) {
            std::cout << words[scores[i].second].word
                      << "\t" << scores[i].first << "\n";
        }
    }

    return 0;
}
