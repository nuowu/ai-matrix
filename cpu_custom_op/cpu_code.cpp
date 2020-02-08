// Copyright 2020 Graphcore Ltd.
#include <string>
#include <vector>
#include <iostream>
#include <stdint.h>

// For gradient function
#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>

extern "C" {  
  void cpuCallback(const std::vector<void*>& data,
                   const std::vector<uint32_t>& number_of_elements,
                   std::vector<void*>& outputs,
                   const std::string& name) {

    std::cout << "----Inside cpuCallback----\n";

    float* input1 = static_cast<float*>(data[0]);
    float* input2 = static_cast<float*>(data[1]);
    float* output = static_cast<float*>(outputs[0]);
    int number_of_elements_per_tensor = number_of_elements[0];

    for(int i = 0; i < number_of_elements_per_tensor; i++) {
      *output = (*input1) + (*input2);
      input1++;
      input2++;
      output++;
    }
    
    for(int i = 0; i < number_of_elements.size(); i++) {
      std::cout << "Input Tensor " << i << " has "
                << number_of_elements[0] << " elements:";

      float* ptr = static_cast<float*>(data[i]);
      for (int j = 0; j < number_of_elements[i]; j++) {
        std::cout << *ptr << ", ";
        ptr++;
      }
      std::cout << std::endl;
    }
    
    std::cout << "Output Tensor " << 0 << " has "
              << number_of_elements_per_tensor << " elements:";
    float* ptr = static_cast<float*>(outputs[0]);
    for (int i = 0; i < number_of_elements_per_tensor; i++) {
      std::cout << *ptr << ", ";
      ptr++;
    }

    std::cout << "\n--------------------------\n";


    
  }

  poplar::program::Program cpuCallback_grad(
    poplar::Graph& graph, const std::vector<poplar::Tensor>& gradients,
    const std::vector<poplar::Tensor>& old_outputs,
    const std::vector<poplar::Tensor>& old_inputs,
    std::vector<poplar::Tensor>& outputs, const std::string& debugPrefix) {

    poplar::program::Sequence seq;

    return seq;
  }
  
}
