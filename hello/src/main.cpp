#include <iostream>
#include <filesystem>
#include <cstdlib>
#include <string>
#include <utils.h>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    char* path = std::getenv("PWD");  // Get current Path

    fs::path currentPath(path); // define file system

    std::cout << MAGENTA << "currentPath: " << BLUE << currentPath << "\n" << GREEN; // output currentPath

    uint64_t filenum = 0;

    for (const auto& entry : fs::directory_iterator(currentPath)) { // repeat for each file in currentPath
        if (fs::is_directory(entry.path())) continue;
        std::cout << entry.path().filename() << "\n";
        filenum++;
    }

    std::cout << MAGENTA << "total files num: " << GREEN << filenum << '\n'; // output total filenum

    return 0;
}

// 让生成的可执行文件在任何路径下都能直接运行可以在~/.bashrc中加入字段 PATH=$PATH:/home/wenjiu/project/AI-Edge/hello/build/linux/x86_64/release
// 不过我们一般会将需要直接被运行的用户可执行文件移动到/usr/local/bin文件夹下，这个文件夹默认被添加到PATH环境变量中，所以直接运行hello程序即可
// 注意将程序移动到这个文件夹中需要root权限，在本文件夹下只需要运行xmake install --admin即可
// 如果要卸载程序，则运行xmake uninstall --admin
// 要在Makefile中实现相同的功能也很容易，只需要sudo cp $(BIN) /usr/local/bin即可
