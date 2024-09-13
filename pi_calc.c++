#include <sycl/sycl.hpp>

constexpr int num=999999;
using namespace sycl;

  int main() {
  auto R = range<1>{ num };
  //Create Buffers A and B
  buffer<float> A{ R };
  //Create a device queue
  queue Q(gpu_selector_v);
  
   float dx,total_area,pi;
   total_area = 0.0;
   dx = 1.0/num;   
      
  //# Print the device name
  std::cout << "Device: " << Q.get_device().get_info<info::device::name>() << "\n";

   
      
  Q.submit([&](handler& h) {
    accessor area(A,h,write_only);
    h.parallel_for(R, [=](auto idx) {
       float x = idx*dx;
		float y = sqrt(1-x*x);
		area[idx] = y*dx;
    }); });

 // And the following is back to device code
 host_accessor result(A,read_only);
  for (int i=0; i<num; ++i)
    total_area += result[i];      
  
  
  pi = 4.0*total_area;
  std::cout << "\n\nPi = " << pi << "\n";      
   
  return 0;    
}