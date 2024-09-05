#include <iostream>
#include <cstdlib>
#include <string>

int main(int argc, char** argv) {
    char* path = std::getenv("PWD"); // Get Current path
    
    if (path != nullptr) {
        std::string currentPath(path);
        std::cout << "Currenr Path:" << currentPath << std::endl;
    } else {
        std::cerr << "Cannot Get Current Path" << std::endl;
    }

    return 0;
}

