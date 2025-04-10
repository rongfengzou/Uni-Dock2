//
// Created by Congcong Liu on 24-10-8.
//


#include <iostream>
#include <vector>

Real aaa(Real x){
    return (int) x;
}


int main(){
    std::vector<Real>list_x = {-1.2, -0.6, -0.3, 0, 0.1, 0.7, 1.8};
    for (auto& x : list_x){
        printf("%f --> %f\n", x, aaa(x));
    }

    return 0;
}
