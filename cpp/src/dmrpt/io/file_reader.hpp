//
// Created by Isuru Ranawaka on 5/17/22.
//

#ifndef DISTRIBUTED_MRPT_FILE_READER_H
#define DISTRIBUTED_MRPT_FILE_READER_H

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <map>

using std::filesystem::directory_iterator;

namespace dmrpt{
    class FileReader {
        private:
           int myrank;
        public:
            FileReader(int myrank);
            std::vector<std::string> parseFileNames(std::string path);
            std::vector<std::string> getMyFileNames(std::vector<std::string> names, int num_of_nodes);

            int getRank();


        };
    }
#endif //DISTRIBUTED_MRPT_FILE_READER_H
