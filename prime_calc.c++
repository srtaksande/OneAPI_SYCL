#include <sycl/sycl.hpp>

constexpr int num=100000;
using namespace sycl;

  int main() {
  auto R = range<1>{ num };
  //Create Buffers A and B
  buffer<int> A{ R };
  //Create a device queue
  queue Q(gpu_selector_v);
  int prime_count = 1;
  //# Print the device name
  std::cout << "Device: " << Q.get_device().get_info<info::device::name>() << "\n";
      
  Q.submit([&](handler& h) {
    accessor isprime(A,h,write_only);
    h.parallel_for(R, [=](auto idx) {
     if(idx>2)
     {
        int i = idx;
        int flag = 0;
		for(int j=2;j<i;j++)	
	    	{
		    if((i%j) == 0)
		    {
			    flag = 1;
			    break;
		    }
	    	}
        	if(flag == 0)
        	{
            	isprime[i] = 1;
        	}
            else
            {
                isprime[i] = 0;
            }   
    }
    }); });

 // And the following is back to device code
 host_accessor isprime(A,read_only);
  for (int i=3; i<num; ++i)
        prime_count += isprime[i];
      
  std::cout << "\n Number of Primes = " << prime_count << "\n";      
  return 0;
}