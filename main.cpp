#include <iostream>
#include <torch/torch.h>

using namespace std;

int main(int c, char** argv) {
    torch::Tensor tensor = torch::rand({2, 3});
    std::cout << tensor << std::endl;

    return 80;
}
