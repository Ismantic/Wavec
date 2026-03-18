#include "ft_wav.h"
#include <iostream>
#include <fstream>

// 创建测试数据文件
void create_test_data(const std::string& filename) {
    std::ofstream fout(filename);
    fout << "__label__sports __label__nba Lakers win championship\n";
    fout << "__label__tech __label__apple Apple releases new iPhone\n";
    fout << "__label__politics President gives speech to congress\n";
    fout << "__label__sports __label__soccer Real Madrid wins Champions League\n";
    fout << "__label__tech Microsoft announces new Surface device\n";
    fout << "__label__politics __label__election Election results announced\n";
    fout << "__label__sports Golden State Warriors win NBA finals\n";
    fout << "__label__tech Google releases new AI model\n";
    fout << "__label__politics __label__international UN holds climate summit\n";
    fout << "__label__sports __label__tennis Nadal wins French Open\n";
    fout.close();
}

int main() {
    // 创建测试数据
    const std::string train_file = "test_data.txt";
    create_test_data(train_file);
    
    // 设置并训练模型
    wavec::FastText model;
    model.SetVecSize(20);          // 设置较小的向量维度便于测试
    model.SetWindow(3);            // 设置上下文窗口
    model.SetMinCount(1);          // 设置单词最小计数
    model.SetMinLabelCount(1);     // 设置标签最小计数
    model.SetCores(1);             // 单线程运行便于调试
    model.SetIter(3);              // 训练3轮
    model.SetSample(0);            // 禁用采样
    
    std::cout << "Starting training..." << std::endl;
    try {
        model.Fit(train_file, "model.vec");
        std::cout << "Training completed successfully!" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Training failed: " << e.what() << std::endl;
        return 1;
    }
    
    // 检查输出文件
    std::ifstream fin("model.vec");
    if (fin) {
        std::cout << "Model file created successfully!" << std::endl;
        std::string line;
        std::getline(fin, line);
        std::cout << "Header: " << line << std::endl;
        
        int count = 0;
        while (std::getline(fin, line)) {
            if (++count <= 5) {
                std::cout << "Vector " << count << ": " << line.substr(0, 40) << "..." << std::endl;
            }
        }
    } else {
        std::cerr << "Failed to create model file!" << std::endl;
    }
    
    return 0;
}
