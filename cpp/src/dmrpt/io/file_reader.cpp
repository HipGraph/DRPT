#include "file_reader.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>


dmrpt::FileReader::FileReader(int rank) {
    myrank = rank;
}

std::vector <std::string> dmrpt::FileReader::parseFileNames(std::string path) {
    std::vector <std::string> vec;
    for (const auto &entry: std::filesystem::directory_iterator(path))
        vec.push_back(entry.path());
    return vec;
}

std::vector <std::string>
dmrpt::FileReader::getMyFileNames(std::vector <std::string> names, int num_of_nodes) {
    std::vector <std::string> map;
    for (int i = 0; i < names.size(); i++) {
        if (i % num_of_nodes == myrank) {
            map.push_back(names[i]);
        }
    }
    return map;
}

int dmrpt::FileReader::getRank() {
    return myrank;
}






