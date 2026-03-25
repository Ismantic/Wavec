#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <cmath>
#include <cstdlib>
#include <random>
#include <algorithm>
#include <limits>

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: kmeans <model.vec> <k> [max_iter] [topn]\n"
                  << "  k         Number of clusters\n"
                  << "  max_iter  Max iterations (default: 50)\n"
                  << "  topn      Print top N words per cluster (default: 20)\n";
        return 1;
    }

    std::string model_file = argv[1];
    int k = std::atoi(argv[2]);
    int max_iter = argc >= 4 ? std::atoi(argv[3]) : 50;
    int topn = argc >= 5 ? std::atoi(argv[4]) : 20;

    // Load vectors
    std::ifstream fin(model_file);
    int vocab_size, dim;
    fin >> vocab_size >> dim;

    std::vector<std::string> words(vocab_size);
    std::vector<float> vecs(static_cast<size_t>(vocab_size) * dim);

    std::cerr << "Loading " << vocab_size << " words, dim=" << dim << "...\n";

    for (int i = 0; i < vocab_size; i++) {
        fin >> words[i];
        float norm = 0;
        for (int j = 0; j < dim; j++) {
            fin >> vecs[i * dim + j];
            norm += vecs[i * dim + j] * vecs[i * dim + j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (int j = 0; j < dim; j++) vecs[i * dim + j] /= norm;
        }
    }

    std::cerr << "Running K-means (k=" << k << ", max_iter=" << max_iter << ")...\n";

    std::vector<float> centroids(static_cast<size_t>(k) * dim, 0);
    std::vector<int> assign(vocab_size);
    std::mt19937 rng(42);

    // Round-robin initialization: word i -> cluster i%k
    for (int i = 0; i < vocab_size; i++)
        assign[i] = i % k;

    // Compute initial centroids
    {
        std::vector<int> counts(k, 0);
        for (int i = 0; i < vocab_size; i++) {
            int c = assign[i];
            counts[c]++;
            for (int j = 0; j < dim; j++)
                centroids[c * dim + j] += vecs[i * dim + j];
        }
        for (int c = 0; c < k; c++) {
            float norm = 0;
            for (int j = 0; j < dim; j++) {
                centroids[c * dim + j] /= counts[c];
                norm += centroids[c * dim + j] * centroids[c * dim + j];
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (int j = 0; j < dim; j++) centroids[c * dim + j] /= norm;
            }
        }
    }

    // Iterate
    for (int it = 0; it < max_iter; it++) {
        // Assign (cosine similarity)
        int changed = 0;
        for (int i = 0; i < vocab_size; i++) {
            float best_sim = -2;
            int best_c = 0;
            for (int c = 0; c < k; c++) {
                float dot = 0;
                for (int j = 0; j < dim; j++)
                    dot += vecs[i * dim + j] * centroids[c * dim + j];
                if (dot > best_sim) {
                    best_sim = dot;
                    best_c = c;
                }
            }
            if (assign[i] != best_c) {
                assign[i] = best_c;
                changed++;
            }
        }

        // Update centroids
        std::vector<int> counts(k, 0);
        std::fill(centroids.begin(), centroids.end(), 0);
        for (int i = 0; i < vocab_size; i++) {
            int c = assign[i];
            counts[c]++;
            for (int j = 0; j < dim; j++)
                centroids[c * dim + j] += vecs[i * dim + j];
        }

        // Handle empty clusters: steal a random member from the largest cluster
        for (int c = 0; c < k; c++) {
            if (counts[c] > 0) continue;
            int largest = std::max_element(counts.begin(), counts.end()) - counts.begin();
            // Find a random member of the largest cluster
            std::uniform_int_distribution<int> pick(0, vocab_size - 1);
            int donor;
            do { donor = pick(rng); } while (assign[donor] != largest);
            assign[donor] = c;
            counts[largest]--;
            counts[c] = 1;
            // Reset centroid accumulators for both clusters
            std::fill(centroids.begin() + largest * dim, centroids.begin() + (largest + 1) * dim, 0);
            for (int i = 0; i < vocab_size; i++) {
                if (assign[i] == largest) {
                    for (int j = 0; j < dim; j++)
                        centroids[largest * dim + j] += vecs[i * dim + j];
                }
            }
            for (int j = 0; j < dim; j++)
                centroids[c * dim + j] = vecs[donor * dim + j];
        }

        for (int c = 0; c < k; c++) {
            if (counts[c] == 0) continue;
            float norm = 0;
            for (int j = 0; j < dim; j++) {
                centroids[c * dim + j] /= counts[c];
                norm += centroids[c * dim + j] * centroids[c * dim + j];
            }
            norm = std::sqrt(norm);
            if (norm > 0) {
                for (int j = 0; j < dim; j++) centroids[c * dim + j] /= norm;
            }
        }

        std::cerr << "Iter " << it + 1 << ": " << changed << " changed\n";
        if (changed == 0) break;
    }

    // Output: for each cluster, print top N words (closest to centroid)
    for (int c = 0; c < k; c++) {
        std::vector<std::pair<float, int>> members;
        for (int i = 0; i < vocab_size; i++) {
            if (assign[i] != c) continue;
            float dot = 0;
            for (int j = 0; j < dim; j++)
                dot += vecs[i * dim + j] * centroids[c * dim + j];
            members.emplace_back(dot, i);
        }
        std::sort(members.begin(), members.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        int n = std::min(topn, static_cast<int>(members.size()));
        std::cout << "=== Cluster " << c << " (" << members.size() << " words) ===\n";
        for (int i = 0; i < n; i++) {
            std::cout << "  " << words[members[i].second] << "\t" << members[i].first << "\n";
        }
    }

    return 0;
}
