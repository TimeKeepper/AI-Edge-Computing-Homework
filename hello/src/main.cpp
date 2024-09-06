#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <string>
#include <utils.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    char* path = std::getenv("PWD"); 

    fs::path currentPath(path);

    std::cout << MAGENTA << "currentPath: " << BLUE << currentPath << "\n" << GREEN;

    uint64_t filenum = 0;

    for (const auto& entry : fs::directory_iterator(currentPath)) {
        if (fs::is_directory(entry.path())) continue;
        std::cout << entry.path().filename() << "\n";
        filenum++;
    }

    std::cout << MAGENTA << "total files num: " << GREEN << filenum << '\n';

    return 0;
}

