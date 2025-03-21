//
// Created by Congcong Liu on 24-10-9.
//

#include "myio.h"

std::string genOutFilePathForMol(const std::string& fn, const std::string& dp_out) {
    std::string tmp = fn;
    if (tmp.size() >= 6 && tmp.substr(tmp.size() - 6, 6) == ".pdbqt") {
        tmp.resize(tmp.size() - 6);
        tmp = tmp + "_out.pdbqt";
    } else if (tmp.size() >= 4 && tmp.substr(tmp.size() - 4, 4) == ".sdf") {
        tmp.resize(tmp.size() - 4);
        tmp = tmp + "_out.sdf";
    }

    if (dp_out=="") {
        return tmp;
    }

    return dp_out + separator() + tmp;
}



void my_info(const std::string& fn){
    std::cout << fn << std::endl << std::flush;
}
