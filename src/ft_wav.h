#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <unordered_map>
#include <algorithm>
#include <chrono>
#include <thread>
#include <random>
#include <cmath>
#include <cassert>

namespace wavec {

std::vector<std::string> StrSplit(const std::string& str, char r = ' ') {
    std::vector<std::string> tokens;
    std::stringstream ss(str);
    std::string token;
    while (std::getline(ss, token, r)) {
        if (!token.empty()) {
            tokens.push_back(token);
        }
    }
    return tokens;
}

class FastText {
private:
    struct Token {
        std::string w;
        uint64_t cn = 0;
        bool is_label = false; // 标记是否为标签
        std::vector<int> point;
        std::vector<char> code;

        Token() = default;
        Token(const std::string& w, bool is_label = false) 
            : w(w), cn(1), is_label(is_label) {}
    };

    struct Document {
        std::vector<std::string> words;
        std::vector<std::string> labels;
    };

    int window = 2;
    int min_count = 1;
    int min_label_count = 1; // 标签最小计数
    int num_core = 1;
    uint64_t dict_size = 0;
    uint64_t vec_size = 50;
    uint64_t token_count = 0;
    int iter = 1;
    float start_alpha = 0.025;
    float sample = 1e-3;

    std::string data_file;
    std::string vec_file;
    std::vector<Token> dict;
    std::unordered_map<std::string, int> dict_map;
    std::vector<float> syn0;  // 输入向量
    std::vector<float> syn1;  // 输出向量
    std::vector<Document> data;

    void LoadData() {
        std::ifstream fin(data_file);
        std::string line;

        std::unordered_map<std::string, uint64_t> word_count;
        std::unordered_map<std::string, uint64_t> label_count_map;

        token_count = 0;

        while (std::getline(fin, line)) {
            if (!line.empty()) {
                Document doc;
                auto tokens = StrSplit(line, ' ');
                for (const auto& token : tokens) {
                    if (token.substr(0, 9) == "__label__") {
                        doc.labels.push_back(token);
                        label_count_map[token]++;
                    } else if (!token.empty()) {
                        doc.words.push_back(token);
                        word_count[token]++;
                        token_count++;
                    }
                }
                if (!doc.words.empty() && !doc.labels.empty()) {
                    data.push_back(doc);
                }
            }
        }

        // 添加单词到字典
        for (const auto& p : word_count) {
            if (p.second >= min_count) {
                dict.push_back(Token(p.first));
                dict.back().cn = p.second;
                dict_map[p.first] = dict_size++;
            } else {
                token_count -= p.second;
            }
        }

        // 添加标签到字典
        for (const auto& p : label_count_map) {
            if (p.second >= min_label_count) {
                dict.push_back(Token(p.first, true));
                dict.back().cn = p.second;
                dict_map[p.first] = dict_size++;
            }
        }

        // 按频率排序
        std::sort(dict.begin(), dict.end(), 
                  [](const Token& a, const Token& b) {
                    return a.cn > b.cn;
                  });
        
        // 重建映射
        dict_map.clear();
        for (int i = 0; i < dict_size; i++) {
            dict_map[dict[i].w] = i;
        }
    }

    void SaveModel() {
        std::ofstream fout(vec_file);
        fout << dict_size << " " << vec_size << "\n";
        for (int i = 0; i < dict_size; i++) {
            fout << dict[i].w << " ";
            for (int c = 0; c < vec_size; c++) {
                fout << syn0[i*vec_size + c] << " ";
            }
            fout << "\n";
        }
        
        // 保存分类层参数
        std::ofstream fout_syn1(vec_file + ".syn1");
        for (int i = 0; i < dict_size; i++) {
            for (int c = 0; c < vec_size; c++) {
                fout_syn1 << syn1[i*vec_size + c] << " ";
            }
            fout_syn1 << "\n";
        }
    }

    inline float Sigmoid(float x) {
        if (x > 6) return 1.0F;
        if (x < -6) return 0.0F;
        return 1.0F / (1.0F + std::exp(-x));
    }

    void CreateBinaryTree() {
        if (dict_size <= 1) return;

        std::vector<uint64_t> count(dict_size * 2);
        std::vector<int> binary(dict_size * 2);
        std::vector<int> parent(dict_size * 2);

        for (int a = 0; a < dict_size; a++) {
            count[a] = dict[a].cn;
        }
        for (int a = dict_size; a < dict_size * 2; a++) {
            count[a] = 1e15;
        }

        int pos1 = dict_size - 1;
        int pos2 = dict_size;

        for (int a = 0; a < dict_size - 1; a++) {
            int min1i, min2i;

            if (pos1 >= 0 && count[pos1] < count[pos2]) {
                min1i = pos1--;
            } else {
                min1i = pos2++;
            }

            if (pos1 >= 0 && count[pos1] < count[pos2]) {
                min2i = pos1--;
            } else {
                min2i = pos2++;
            }

            count[dict_size + a] = count[min1i] + count[min2i];
            parent[min1i] = dict_size + a;
            parent[min2i] = dict_size + a;
            binary[min2i] = 1;
        }

        for (int a = 0; a < dict_size; a++) {
            int b = a;
            std::vector<char> code;
            std::vector<int> point;

            while (b != dict_size * 2 - 2) {
                code.push_back(binary[b]);
                point.push_back(b);
                b = parent[b];
            }

            int n = code.size();
            dict[a].code.resize(n);
            dict[a].point.resize(n);

            for (int i = 0; i < n; i++) {
                dict[a].code[n - 1 - i] = code[i];
                dict[a].point[n - 1 - i] = point[i] - dict_size;
            }
        }
    }

    void InitNet() {
        syn0.resize(dict_size * vec_size, 0.0F);
        syn1.resize(dict_size * vec_size, 0.0F);

        std::mt19937_64 rng{std::random_device{}()};
        std::uniform_real_distribution<float> dist(-0.5F/vec_size, 0.5F/vec_size);
        for (size_t i = 0; i < dict_size * vec_size; i++) {
            syn0[i] = dist(rng);
            syn1[i] = 0.0F;
        }

        CreateBinaryTree();
    }

    // 单词上下文训练
    void CBOW(float alpha, int t, const std::vector<int>& context) {
        if (context.empty()) return;
        
        std::vector<float> neu1(vec_size, 0.0F);
        std::vector<float> neu1e(vec_size, 0.0F);

        // 平均上下文向量
        for (int i : context) {
            for (int c = 0; c < vec_size; c++) {
                neu1[c] += syn0[i * vec_size + c];
            }
        }
        for (int c = 0; c < vec_size; c++) {
            neu1[c] /= context.size();
        }

        // 层次Softmax
        for (int v = 0; v < dict[t].code.size(); v++) {
            int point_index = dict[t].point[v];
            // 确保point_index在有效范围内
            if (point_index < 0 || point_index >= static_cast<int>(dict_size)) {
                continue;
            }
            
            int l2 = point_index * vec_size;
            float f = 0;
            for (int c = 0; c < vec_size; c++) {
                f += neu1[c] * syn1[c + l2];
            }

            float p = Sigmoid(f);
            float g = (1 - dict[t].code[v] - p) * alpha;

            for (int c = 0; c < vec_size; c++) {
                neu1e[c] += g * syn1[c + l2];
                syn1[c + l2] += g * neu1[c];
            }
        }

        // 更新输入向量
        for (int i : context) {
            for (int c = 0; c < vec_size; c++) {
                syn0[i * vec_size + c] += neu1e[c] / context.size();
            }
        }
    }

    // 文档分类训练
    void TrainDocument(float alpha, const std::vector<int>& word_indexes,
                      const std::vector<int>& label_indexes) {
        if (word_indexes.empty() || label_indexes.empty()) return;
        
        std::vector<float> neu1(vec_size, 0.0F);
        std::vector<float> neu1e(vec_size, 0.0F);

        // 计算文档平均向量
        for (int idx : word_indexes) {
            // 确保索引有效
            if (idx < 0 || idx >= static_cast<int>(dict_size)) continue;
            
            for (int c = 0; c < vec_size; c++) {
                neu1[c] += syn0[idx * vec_size + c];
            }
        }
        for (int c = 0; c < vec_size; c++) {
            neu1[c] /= word_indexes.size();
        }

        // 对每个标签进行训练
        for (int label_idx : label_indexes) {
            // 确保标签索引有效
            if (label_idx < 0 || label_idx >= static_cast<int>(dict_size)) continue;
            
            std::fill(neu1e.begin(), neu1e.end(), 0.0F);
            
            // 层次Softmax
            for (int v = 0; v < dict[label_idx].code.size(); v++) {
                int point_index = dict[label_idx].point[v];
                // 确保point_index在有效范围内
                if (point_index < 0 || point_index >= static_cast<int>(dict_size)) {
                    continue;
                }
                
                int l2 = point_index * vec_size;
                float f = 0;
                for (int c = 0; c < vec_size; c++) {
                    f += neu1[c] * syn1[c + l2];
                }

                float p = Sigmoid(f);
                float g = (1 - dict[label_idx].code[v] - p) * alpha;

                for (int c = 0; c < vec_size; c++) {
                    neu1e[c] += g * syn1[c + l2];
                    syn1[c + l2] += g * neu1[c];
                }
            }

            // 更新单词向量
            for (int idx : word_indexes) {
                // 再次确保索引有效
                if (idx < 0 || idx >= static_cast<int>(dict_size)) continue;
                
                for (int c = 0; c < vec_size; c++) {
                    syn0[idx * vec_size + c] += neu1e[c] / word_indexes.size();
                }
            }
        }
    }

    bool ShouldDiscard(std::mt19937_64& rng, uint64_t count) {
        if (sample <= 0) return false;
        float freq = static_cast<float>(count) / token_count;
        float keep_prob = std::sqrt(sample / freq) + sample / freq;
        std::uniform_real_distribution<float> dist(0.0F, 1.0F);
        return dist(rng) > keep_prob;
    }

    void FitOne(int core) {
        std::mt19937_64 rng(std::random_device{}() + core);
        float alpha = start_alpha;

        uint64_t count = 0;
        uint64_t docs_one_core = data.size() / num_core;
        uint64_t start_doc = core * docs_one_core;
        uint64_t end_doc = (core == num_core - 1) ? data.size() : start_doc + docs_one_core;

        std::uniform_int_distribution<int> window_dist(1, window);

        uint64_t core_count = 0;
        for (uint64_t doc = start_doc; doc < end_doc; doc++) {
            core_count += data[doc].words.size();
        }

        for (int iteration = 0; iteration < iter; iteration++) {
            for (uint64_t doc_idx = start_doc; doc_idx < end_doc; doc_idx++) {
                const auto& doc = data[doc_idx];
                const auto& ws = doc.words;

                // 单词上下文训练
                for (int pos = 0; pos < ws.size(); pos++) {
                    if (++count % 10000 == 0) {
                        float progress = static_cast<float>(count) / (core_count * iter);
                        alpha = start_alpha * (1 - progress);
                        alpha = std::max(alpha, 0.0001f);
                    }

                    auto it = dict_map.find(ws[pos]);
                    if (it == dict_map.end()) continue;

                    int t = it->second;

                    // 跳过标签
                    if (dict[t].is_label) continue;

                    // 采样
                    if (sample > 0 && ShouldDiscard(rng, dict[t].cn)) {
                        continue;
                    }

                    std::vector<int> context;
                    int current_window = window_dist(rng);

                    for (int a = -current_window; a <= current_window; a++) {
                        if (a == 0) continue;

                        int context_pos = pos + a;
                        if (context_pos >= 0 && context_pos < ws.size()) {
                            auto ctx_it = dict_map.find(ws[context_pos]);
                            if (ctx_it != dict_map.end() && !dict[ctx_it->second].is_label) {
                                context.push_back(ctx_it->second);
                            }
                        }
                    }

                    if (!context.empty()) {
                        CBOW(alpha, t, context);
                    }
                }

                // 文档分类训练
                std::vector<int> word_indexes;
                std::vector<int> label_indexes;

                // 收集文档中的单词索引
                for (const auto& w : doc.words) {
                    auto it = dict_map.find(w);
                    if (it != dict_map.end() && !dict[it->second].is_label) {
                        word_indexes.push_back(it->second);
                    }
                }

                // 收集文档中的标签索引
                for (const auto& label : doc.labels) {
                    auto it = dict_map.find(label);
                    if (it != dict_map.end() && dict[it->second].is_label) {
                        label_indexes.push_back(it->second);
                    }
                }

                // 训练文档分类
                if (!word_indexes.empty() && !label_indexes.empty()) {
                    TrainDocument(alpha, word_indexes, label_indexes);
                }
            }
        }
    }

public:
    FastText() = default;

    void Fit(const std::string& data_filename, const std::string& vec_filename) {
        data_file = data_filename;
        vec_file = vec_filename;

        LoadData();
        if (dict_size == 0) {
            std::cerr << "Error: No valid data loaded. Check min_count and min_label_count settings." << std::endl;
            return;
        }
        InitNet();

        auto start = std::chrono::steady_clock::now();

        std::vector<std::thread> threads;
        for (int i = 0; i < num_core; i++) {
            threads.emplace_back(&FastText::FitOne, this, i);
        }

        for (auto& thread : threads) {
            thread.join();
        }

        auto end = std::chrono::steady_clock::now();
        std::chrono::duration<float> elapsed = end - start;
        std::cout << "Fit Done! Time: " << elapsed.count() << "s" << std::endl;

        SaveModel();
    }
    
    void SetVecSize(int size) { vec_size = size; }
    void SetWindow(int w) { window = w; }
    void SetMinCount(int min) { min_count = min; }
    void SetMinLabelCount(int min) { min_label_count = min; }
    void SetCores(int n) { num_core = n; }
    void SetIter(int n) { iter = n; }
    void SetSample(float t) { sample = t; }
};

} // namespace wavec
