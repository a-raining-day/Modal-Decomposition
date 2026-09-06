/***********************************************************
* Smithsonian Astrophysical Observatory
* Submillimeter Receiver Laboratory
* am
*
* transform.h                   S. Paine rev. 2014 August 26
*
* Declarations for transform.c
************************************************************/

/***********************************************************
* Third-party notice: these transform routines originate from
* the Smithsonian Astrophysical Observatory (SAO), Submillimeter
* Receiver Laboratory, "am" project (S. Paine), and were
* acquired via
*   https://github.com/waddafunk/Smithsonians_Discrete_Hilbert_Fourier_Hartley_Transforms
* Free to use, with attribution and credit to the Smithsonian
* Astrophysical Observatory (see the full license notice in the
* project README "Acknowledgement" section).
* Hilbert-transform phase convention: +90 deg, NOT the -90 deg
* of MATLAB/SciPy. To get the MATLAB-compatible result multiply
* the orthogonal component by exp(j*PI) = -1, i.e. multiply the
* imaginary part by -1 for a real input (matlab_phase=True).
************************************************************/

#ifndef AM_TRANSFORM_H
#define AM_TRANSFORM_H

void fft_dif(double*, unsigned long);
void ifft_dit(double*, unsigned long);
void fht_dif(double*, unsigned long);
void fht_dit(double*, unsigned long);
void hilbert(double*, unsigned long);
void bitrev_permute(double*, unsigned long);
void bitrev_permute_real(double*, unsigned long);
void ft_benchmarks(void);

#endif /* AM_TRANSFORM_H */
