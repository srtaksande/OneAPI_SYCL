#include <sycl/sycl.hpp>

constexpr int num=100;
using namespace sycl;

  int main() {
  auto R = range<1>{ num };
  auto RR = range<1>{ num*num };
  //Create Buffers A and B
  buffer<int> A{ RR }, B{ R }, C{ R };
  //Create a device queue
  queue Q(gpu_selector_v);

  //# Print the device name
  std::cout << "Device: " << Q.get_device().get_info<info::device::name>() << "\n";

  
  // Initialize the data    
  Q.submit([&](handler& h) {
    accessor outa(A,h,write_only);
    accessor outb(B,h,write_only);
    h.parallel_for(R, [=](auto idx) {
      for(int i=0;i<num;i++)
        outa[i*num+idx] = 1;
        
      outb[idx] = 1;
    }); });
   
      
  Q.submit([&](handler& h) {
    accessor outa(A,h,read_only);
    accessor outb(B,h,read_only);
    accessor outc(C,h,write_only);
    h.parallel_for(R, [=](auto idx) {
      int sum = 0;
		for(int j=0;j<num;j++)
		{
			sum = sum + outa[idx*num+j]*outb[idx];	
		}
		outc[idx] =  sum;  
    }); });

 // And the following is back to device code
 host_accessor result(C,read_only);
  for (int i=0; i<num; ++i)
    std::cout << result[i] << "\n";      
  return 0;
}