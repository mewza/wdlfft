# wdlfft
templated C++ wrapper (capable of intrinsic vector type) FFT based on WDL FFT original by  D. J. Bernstein

it works. I use it daily.

Usage example:


#include <stdio.h>
#include <stdlib.h>
#include "wdlfft.h"

DECL_WDLFFT(simd_double8)

int main() 
{
  // vector data
  simd_double8 data[1024];
  
  // initialize WDLFFT for simd_double8
  WDLFFT<simd_double8>::InitFFTData(1024);

  // instantiate WDLFFT
  WDLFFT<simd_double8> wdl;

 // data is waveform
  wdl.real_fft(1024, data, 0);

 // data is now spectrum
  wdl.real_fft(1024, data, 1);

  // data is back to waveform don't forget to scale to 1./1024.th 
  // PARTY!
}
