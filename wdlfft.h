/*
 **  This is a C++ templated intrinsic vector-compatible
 **  wrapper for the Berstein's WDL FFT
 **  Original WDL FFT routine Copyright 1999 D. J. Bernstein
 **
 **  C++ wrapper (C)2024 Dmitry Boldyrev
 **  GITHUB: https://github.com/mewza or Email: subband@protonmail.com
 **  LICENSE: Wrapper is FREE to use in commercial but would
 **  appreciate a hello in the credits. Cruise over to my GITHUB page
 **  and you may find other goodies useful.
 **
 **  This file implements the WDL FFT library. These routines are based on the
 **  DJBFFT library, which are   Copyright 1999 D. J. Bernstein, djb@pobox.com
 **
 **  The DJB FFT web page is:  http://cr.yp.to/djbfft.html
 */

#pragma once

#include <math.h>
#include <string.h>
#include <simd/simd.h>

#ifndef CMPLX_T_TYPE
#define CMPLX_T_TYPE

template <typename T>
struct cmplxT {
    T re;
    T im;
};

#endif // CMPLX_T_TYPE

#define FFT_MINBITLEN           4  // 16 min
#define FFT_MAXBITLEN           15 // 32768 max
#define FFT_MINBITLEN_REORDER   (FFT_MINBITLEN-1)

// #define WDL_FFT_NO_PERMUTE

template <typename T>
class WDLFFT {
public:
    
    static constexpr int floorlog2(int x) {
        return (x == 1) ? 0 : 1 + floorlog2(x >> 1);
    }
    static const int S_TAB_SIZE = (2 << FFT_MAXBITLEN) + 24 * (FFT_MAXBITLEN - FFT_MINBITLEN_REORDER + 1);
    static const int IDXPERM_SIZE = 2 << FFT_MAXBITLEN;
    
    /*
     * NOTE: Must call this once per C++ template "T" <type> type in your main()
     *       and also DECL_WDLFFT(<type>) to declare globals for that type
     */
    static void InitFFTData(int fftsize)
    {
        // fprintf(stderr, "InitFFTData( %d ), x: %d\n", fftsize, x);
        WDL_fft_init();

        int n = floorlog2(fftsize);
        fft_make_reorder_table(n, fft_reorder_table_for_bitsize(n));
    }
    
    /* 
     * Expects double input[0..len-1] scaled by 0.5/len, returns
     * cmplxT<T> output[0..len/2-1], for len >= 4 order by
     * WDL_fft_permute(len/2). Note that output[len/2].re is stored in
     * output[0].im.
     */

    static void real_fft(T* buf, int32_t len, int32_t isInverse)
    {
        switch (len)
        {
            case 2: if (!isInverse) r2(buf); else v2(buf); break;
            case 4: case 8: two_for_one(buf, 0, len, isInverse); break;
#define TMP(x) case x: two_for_one(buf, d##x, len, isInverse); break;
                TMP(16)
                TMP(32)
                TMP(64)
                TMP(128)
                TMP(256)
                TMP(512)
                TMP(1024)
                TMP(2048)
                TMP(4096)
                TMP(8192)
                TMP(16384)
                TMP(32768)
#undef TMP
        }
    }
    
    static cmplxT<T> d16[3];
    static cmplxT<T> d32[7];
    static cmplxT<T> d64[15];
    static cmplxT<T> d128[31];
    static cmplxT<T> d256[63];
    static cmplxT<T> d512[127];
    static cmplxT<T> d1024[127];
    static cmplxT<T> d2048[255];
    static cmplxT<T> d4096[511];
    static cmplxT<T> d8192[1023];
    static cmplxT<T> d16384[2047];
    static cmplxT<T> d32768[4095];

    static int32_t s_tab[S_TAB_SIZE]; // big 256kb table, ugh
    static int32_t _idxperm[IDXPERM_SIZE];
    
    #define sqrthalf (d16[1].re)
        
    #define VOL *(volatile T *)&
        
    #define TRANSFORM(a0,a1,a2,a3,wre,wim) { \
    t6 = a2.re; \
    t1 = a0.re - t6; \
    t6 += a0.re; \
    a0.re = t6; \
    t3 = a3.im; \
    t4 = a1.im - t3; \
    t8 = t1 - t4; \
    t1 += t4; \
    t3 += a1.im; \
    a1.im = t3; \
    t5 = wre; \
    t7 = t8 * t5; \
    t4 = t1 * t5; \
    t8 *= wim; \
    t2 = a3.re; \
    t3 = a1.re - t2; \
    t2 += a1.re; \
    a1.re = t2; \
    t1 *= wim; \
    t6 = a2.im; \
    t2 = a0.im - t6; \
    t6 += a0.im; \
    a0.im = t6; \
    t6 = t2 + t3; \
    t2 -= t3; \
    t3 = t6 * wim; \
    t7 -= t3; \
    a2.re = t7; \
    t6 *= t5; \
    t6 += t8; \
    a2.im = t6; \
    t5 *= t2; \
    t5 -= t1; \
    a3.im = t5; \
    t2 *= wim; \
    t4 += t2; \
    a3.re = t4; \
    }
        
    #define TRANSFORMHALF(a0,a1,a2,a3) { \
    t1 = a2.re; \
    t5 = a0.re - t1; \
    t1 += a0.re; \
    a0.re = t1; \
    t4 = a3.im; \
    t8 = a1.im - t4; \
    t1 = t5 - t8; \
    t5 += t8; \
    t4 += a1.im; \
    a1.im = t4; \
    t3 = a3.re; \
    t7 = a1.re - t3; \
    t3 += a1.re; \
    a1.re = t3; \
    t8 = a2.im; \
    t6 = a0.im - t8; \
    t2 = t6 + t7; \
    t6 -= t7; \
    t8 += a0.im; \
    a0.im = t8; \
    t4 = t6 + t5; \
    t3 = sqrthalf; \
    t4 *= t3; \
    a3.re = t4; \
    t6 -= t5; \
    t6 *= t3; \
    a3.im = t6; \
    t7 = t1 - t2; \
    t7 *= t3; \
    a2.re = t7; \
    t2 += t1; \
    t2 *= t3; \
    a2.im = t2; \
    }
        
    #define TRANSFORMZERO(a0,a1,a2,a3) { \
    t5 = a2.re; \
    t1 = a0.re - t5; \
    t5 += a0.re; \
    a0.re = t5; \
    t8 = a3.im; \
    t4 = a1.im - t8; \
    t7 = a3.re; \
    t6 = t1 - t4; \
    a2.re = t6; \
    t1 += t4; \
    a3.re = t1; \
    t8 += a1.im; \
    a1.im = t8; \
    t3 = a1.re - t7; \
    t7 += a1.re; \
    a1.re = t7; \
    t6 = a2.im; \
    t2 = a0.im - t6; \
    t7 = t2 + t3; \
    a2.im = t7; \
    t2 -= t3; \
    a3.im = t2; \
    t6 += a0.im; \
    a0.im = t6; \
    }
        
    #define UNTRANSFORM(a0,a1,a2,a3,wre,wim) { \
    t6 = VOL wre; \
    t1 = VOL a2.re; \
    t1 *= t6; \
    t8 = VOL wim; \
    t3 = VOL a2.im; \
    t3 *= t8; \
    t2 = VOL a2.im; \
    t4 = VOL a2.re; \
    t5 = VOL a3.re; \
    t5 *= t6; \
    t7 = VOL a3.im; \
    t1 += t3; \
    t7 *= t8; \
    t5 -= t7; \
    t3 = t5 + t1; \
    t5 -= t1; \
    t2 *= t6; \
    t6 *= a3.im; \
    t4 *= t8; \
    t2 -= t4; \
    t8 *= a3.re; \
    t6 += t8; \
    t1 = a0.re - t3; \
    t3 += a0.re; \
    a0.re = t3; \
    t7 = a1.im - t5; \
    t5 += a1.im; \
    a1.im = t5; \
    t4 = t2 - t6; \
    t6 += t2; \
    t8 = a1.re - t4; \
    t4 += a1.re; \
    a1.re = t4; \
    t2 = a0.im - t6; \
    t6 += a0.im; \
    a0.im = t6; \
    a2.re = t1; \
    a3.im = t7; \
    a3.re = t8; \
    a2.im = t2; \
    }
        
        
    #define UNTRANSFORMHALF(a0,a1,a2,a3) { \
    t6 = sqrthalf; \
    t1 = a2.re; \
    t2 = a2.im - t1; \
    t2 *= t6; \
    t1 += a2.im; \
    t1 *= t6; \
    t4 = a3.im; \
    t3 = a3.re - t4; \
    t3 *= t6; \
    t4 += a3.re; \
    t4 *= t6; \
    t8 = t3 - t1; \
    t7 = t2 - t4; \
    t1 += t3; \
    t2 += t4; \
    t4 = a1.im - t8; \
    a3.im = t4; \
    t8 += a1.im; \
    a1.im = t8; \
    t3 = a1.re - t7; \
    a3.re = t3; \
    t7 += a1.re; \
    a1.re = t7; \
    t5 = a0.re - t1; \
    a2.re = t5; \
    t1 += a0.re; \
    a0.re = t1; \
    t6 = a0.im - t2; \
    a2.im = t6; \
    t2 += a0.im; \
    a0.im = t2; \
    }
        
    #define UNTRANSFORMZERO(a0,a1,a2,a3) { \
    t2 = a3.im; \
    t3 = a2.im - t2; \
    t2 += a2.im; \
    t1 = a2.re; \
    t4 = a3.re - t1; \
    t1 += a3.re; \
    t5 = a0.re - t1; \
    a2.re = t5; \
    t6 = a0.im - t2; \
    a2.im = t6; \
    t7 = a1.re - t3; \
    a3.re = t7; \
    t8 = a1.im - t4; \
    a3.im = t8; \
    t1 += a0.re; \
    a0.re = t1; \
    t2 += a0.im; \
    a0.im = t2; \
    t3 += a1.re; \
    a1.re = t3; \
    t4 += a1.im; \
    a1.im = t4; \
    }
    
    void reorder_buffer(int sz, T *buf, int isInverse)
    {
        cmplxT<T> *data = (cmplxT<T>*)buf;
        int bitsz = floorlog2(sz);
        const int32_t *tab = fft_reorder_table_for_bitsize(bitsz);
        if (isInverse)
        {
            while (*tab)
            {
                const int32_t sidx = *tab++;
                cmplxT<T> a = data[sidx];
                for (;;)
                {
                    cmplxT<T> ta;
                    const int32_t idx = *tab++;
                    if (!idx) break;
                    ta = data[idx];
                    data[idx] = a;
                    a = ta;
                }
                data[sidx] = a;
            }
        } else
        {
            while (*tab)
            {
                const int32_t sidx = *tab++;
                int32_t lidx = sidx;
                const cmplxT<T> sta = data[lidx];
                for (;;)
                {
                    const int32_t idx = *tab++;
                    if (!idx) break;
                    data[lidx] = data[idx];
                    lidx = idx;
                }
                data[lidx] = sta;
            }
        }
    }
    
    static void c2(cmplxT<T> *a)
    {
        T t1;
        
        t1 = a[1].re;
        a[1].re = a[0].re - t1;
        a[0].re += t1;
        
        t1 = a[1].im;
        a[1].im = a[0].im - t1;
        a[0].im += t1;
    }
    
    static inline void c4(cmplxT<T> *a)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        
        t5 = a[2].re;
        t1 = a[0].re - t5;
        t7 = a[3].re;
        t5 += a[0].re;
        t3 = a[1].re - t7;
        t7 += a[1].re;
        t8 = t5 + t7;
        a[0].re = t8;
        t5 -= t7;
        a[1].re = t5;
        t6 = a[2].im;
        t2 = a[0].im - t6;
        t6 += a[0].im;
        t5 = a[3].im;
        a[2].im = t2 + t3;
        t2 -= t3;
        a[3].im = t2;
        t4 = a[1].im - t5;
        a[3].re = t1 + t4;
        t1 -= t4;
        a[2].re = t1;
        t5 += a[1].im;
        a[0].im = t6 + t5;
        t6 -= t5;
        a[1].im = t6;
    }
    
    static void c8(cmplxT<T> *a)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        
        t7 = a[4].im;
        t4 = a[0].im - t7;
        t7 += a[0].im;
        a[0].im = t7;
        
        t8 = a[6].re;
        t5 = a[2].re - t8;
        t8 += a[2].re;
        a[2].re = t8;
        
        t7 = a[6].im;
        a[6].im = t4 - t5;
        t4 += t5;
        a[4].im = t4;
        
        t6 = a[2].im - t7;
        t7 += a[2].im;
        a[2].im = t7;
        
        t8 = a[4].re;
        t3 = a[0].re - t8;
        t8 += a[0].re;
        a[0].re = t8;
        
        a[4].re = t3 - t6;
        t3 += t6;
        a[6].re = t3;
        
        t7 = a[5].re;
        t3 = a[1].re - t7;
        t7 += a[1].re;
        a[1].re = t7;
        
        t8 = a[7].im;
        t6 = a[3].im - t8;
        t8 += a[3].im;
        a[3].im = t8;
        t1 = t3 - t6;
        t3 += t6;
        
        t7 = a[5].im;
        t4 = a[1].im - t7;
        t7 += a[1].im;
        a[1].im = t7;
        
        t8 = a[7].re;
        t5 = a[3].re - t8;
        t8 += a[3].re;
        a[3].re = t8;
        
        t2 = t4 - t5;
        t4 += t5;
        
        t6 = t1 - t4;
        t8 = sqrthalf;
        t6 *= t8;
        a[5].re = a[4].re - t6;
        t1 += t4;
        t1 *= t8;
        a[5].im = a[4].im - t1;
        t6 += a[4].re;
        a[4].re = t6;
        t1 += a[4].im;
        a[4].im = t1;
        
        t5 = t2 - t3;
        t5 *= t8;
        a[7].im = a[6].im - t5;
        t2 += t3;
        t2 *= t8;
        a[7].re = a[6].re - t2;
        t2 += a[6].re;
        a[6].re = t2;
        t5 += a[6].im;
        a[6].im = t5;
        
        c4(a);
    }
    
    static void c16(cmplxT<T> *a)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        
        TRANSFORMZERO(a[0],a[4],a[8],a[12]);
        TRANSFORM(a[1],a[5],a[9],a[13],d16[0].re,d16[0].im);
        TRANSFORMHALF(a[2],a[6],a[10],a[14]);
        TRANSFORM(a[3],a[7],a[11],a[15],d16[0].im,d16[0].re);
        c4(a + 8);
        c4(a + 12);
        
        c8(a);
    }
    
    /* a[0...8n-1], w[0...2n-2]; n >= 2 */
    static void cpass(cmplxT<T> *a,const cmplxT<T> *w,uint32_t n)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        cmplxT<T> *a1;
        cmplxT<T> *a2;
        cmplxT<T> *a3;
        
        a2 = a + 4 * n;
        a1 = a + 2 * n;
        a3 = a2 + 2 * n;
        --n;
        
        TRANSFORMZERO(a[0],a1[0],a2[0],a3[0]);
        TRANSFORM(a[1],a1[1],a2[1],a3[1],w[0].re,w[0].im);
        
        for (;;) {
            TRANSFORM(a[2],a1[2],a2[2],a3[2],w[1].re,w[1].im);
            TRANSFORM(a[3],a1[3],a2[3],a3[3],w[2].re,w[2].im);
            if (!--n) break;
            a += 2;
            a1 += 2;
            a2 += 2;
            a3 += 2;
            w += 2;
        }
    }
    
    static void c32(cmplxT<T> *a)
    {
        cpass(a,d32,4);
        c8(a + 16);
        c8(a + 24);
        c16(a);
    }
    
    static void c64(cmplxT<T> *a)
    {
        cpass(a,d64,8);
        c16(a + 32);
        c16(a + 48);
        c32(a);
    }
    
    static void c128(cmplxT<T> *a)
    {
        cpass(a,d128,16);
        c32(a + 64);
        c32(a + 96);
        c64(a);
    }
    
    static void c256(cmplxT<T> *a)
    {
        cpass(a,d256,32);
        c64(a + 128);
        c64(a + 192);
        c128(a);
    }
    
    static void c512(cmplxT<T> *a)
    {
        cpass(a,d512,64);
        c128(a + 384);
        c128(a + 256);
        c256(a);
    }
    
    /* a[0...8n-1], w[0...n-2]; n even, n >= 4 */
    static void cpassbig(cmplxT<T> *a,const cmplxT<T> *w,uint32_t n)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        cmplxT<T> *a1;
        cmplxT<T> *a2;
        cmplxT<T> *a3;
        uint32_t k;
        
        a2 = a + 4 * n;
        a1 = a + 2 * n;
        a3 = a2 + 2 * n;
        k = n - 2;
        
        TRANSFORMZERO(a[0],a1[0],a2[0],a3[0]);
        TRANSFORM(a[1],a1[1],a2[1],a3[1],w[0].re,w[0].im);
        a += 2;
        a1 += 2;
        a2 += 2;
        a3 += 2;
        
        do {
            TRANSFORM(a[0],a1[0],a2[0],a3[0],w[1].re,w[1].im);
            TRANSFORM(a[1],a1[1],a2[1],a3[1],w[2].re,w[2].im);
            a += 2;
            a1 += 2;
            a2 += 2;
            a3 += 2;
            w += 2;
        } while (k -= 2);
        
        TRANSFORMHALF(a[0],a1[0],a2[0],a3[0]);
        TRANSFORM(a[1],a1[1],a2[1],a3[1],w[0].im,w[0].re);
        a += 2;
        a1 += 2;
        a2 += 2;
        a3 += 2;
        
        k = n - 2;
        do {
            TRANSFORM(a[0],a1[0],a2[0],a3[0],w[-1].im,w[-1].re);
            TRANSFORM(a[1],a1[1],a2[1],a3[1],w[-2].im,w[-2].re);
            a += 2;
            a1 += 2;
            a2 += 2;
            a3 += 2;
            w -= 2;
        } while (k -= 2);
    }
    
    
    static void c1024(cmplxT<T> *a)
    {
        cpassbig(a,d1024,128);
        c256(a + 768);
        c256(a + 512);
        c512(a);
    }
    
    static void c2048(cmplxT<T> *a)
    {
        cpassbig(a,d2048,256);
        c512(a + 1536);
        c512(a + 1024);
        c1024(a);
    }
    
    static void c4096(cmplxT<T> *a)
    {
        cpassbig(a,d4096,512);
        c1024(a + 3072);
        c1024(a + 2048);
        c2048(a);
    }
    
    static void c8192(cmplxT<T> *a)
    {
        cpassbig(a,d8192,1024);
        c2048(a + 6144);
        c2048(a + 4096);
        c4096(a);
    }
    
    static void c16384(cmplxT<T> *a)
    {
        cpassbig(a,d16384,2048);
        c4096(a + 8192 + 4096);
        c4096(a + 8192);
        c8192(a);
    }
    
    static void c32768(cmplxT<T> *a)
    {
        cpassbig(a,d32768,4096);
        c8192(a + 16384 + 8192);
        c8192(a + 16384);
        c16384(a);
    }
    
    
    /* n even, n > 0 */
    void WDL_fft_complexmul(cmplxT<T> *a,cmplxT<T> *b,int32_t n)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        if (n<2 || (n&1)) return;
        
        do {
            t1 = a[0].re * b[0].re;
            t2 = a[0].im * b[0].im;
            t3 = a[0].im * b[0].re;
            t4 = a[0].re * b[0].im;
            t5 = a[1].re * b[1].re;
            t6 = a[1].im * b[1].im;
            t7 = a[1].im * b[1].re;
            t8 = a[1].re * b[1].im;
            t1 -= t2;
            t3 += t4;
            t5 -= t6;
            t7 += t8;
            a[0].re = t1;
            a[1].re = t5;
            a[0].im = t3;
            a[1].im = t7;
            a += 2;
            b += 2;
        } while (n -= 2);
    }
    
    void WDL_fft_complexmul2(cmplxT<T> *c, cmplxT<T> *a, cmplxT<T> *b, int32_t n)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        if (n<2 || (n&1)) return;
        
        do {
            t1 = a[0].re * b[0].re;
            t2 = a[0].im * b[0].im;
            t3 = a[0].im * b[0].re;
            t4 = a[0].re * b[0].im;
            t5 = a[1].re * b[1].re;
            t6 = a[1].im * b[1].im;
            t7 = a[1].im * b[1].re;
            t8 = a[1].re * b[1].im;
            t1 -= t2;
            t3 += t4;
            t5 -= t6;
            t7 += t8;
            c[0].re = t1;
            c[1].re = t5;
            c[0].im = t3;
            c[1].im = t7;
            a += 2;
            b += 2;
            c += 2;
        } while (n -= 2);
    }
    void WDL_fft_complexmul3(cmplxT<T> *c, cmplxT<T> *a, cmplxT<T> *b, int32_t n)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        if (n<2 || (n&1)) return;
        
        do {
            t1 = a[0].re * b[0].re;
            t2 = a[0].im * b[0].im;
            t3 = a[0].im * b[0].re;
            t4 = a[0].re * b[0].im;
            t5 = a[1].re * b[1].re;
            t6 = a[1].im * b[1].im;
            t7 = a[1].im * b[1].re;
            t8 = a[1].re * b[1].im;
            t1 -= t2;
            t3 += t4;
            t5 -= t6;
            t7 += t8;
            c[0].re += t1;
            c[1].re += t5;
            c[0].im += t3;
            c[1].im += t7;
            a += 2;
            b += 2;
            c += 2;
        } while (n -= 2);
    }
    
    static inline void u4(cmplxT<T> *a)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        
        t1 = VOL a[1].re;
        t3 = a[0].re - t1;
        t6 = VOL a[2].re;
        t1 += a[0].re;
        t8 = a[3].re - t6;
        t6 += a[3].re;
        a[0].re = t1 + t6;
        t1 -= t6;
        a[2].re = t1;
        
        t2 = VOL a[1].im;
        t4 = a[0].im - t2;
        t2 += a[0].im;
        t5 = VOL a[3].im;
        a[1].im = t4 + t8;
        t4 -= t8;
        a[3].im = t4;
        
        t7 = a[2].im - t5;
        t5 += a[2].im;
        a[1].re = t3 + t7;
        t3 -= t7;
        a[3].re = t3;
        a[0].im = t2 + t5;
        t2 -= t5;
        a[2].im = t2;
    }
    
    static void u8(cmplxT<T> *a)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        
        u4(a);
        
        t1 = a[5].re;
        a[5].re = a[4].re - t1;
        t1 += a[4].re;
        
        t3 = a[7].re;
        a[7].re = a[6].re - t3;
        t3 += a[6].re;
        
        t8 = t3 - t1;
        t1 += t3;
        
        t6 = a[2].im - t8;
        t8 += a[2].im;
        a[2].im = t8;
        
        t5 = a[0].re - t1;
        a[4].re = t5;
        t1 += a[0].re;
        a[0].re = t1;
        
        t2 = a[5].im;
        a[5].im = a[4].im - t2;
        t2 += a[4].im;
        
        t4 = a[7].im;
        a[7].im = a[6].im - t4;
        t4 += a[6].im;
        
        a[6].im = t6;
        
        t7 = t2 - t4;
        t2 += t4;
        
        t3 = a[2].re - t7;
        a[6].re = t3;
        t7 += a[2].re;
        a[2].re = t7;
        
        t6 = a[0].im - t2;
        a[4].im = t6;
        t2 += a[0].im;
        a[0].im = t2;
        
        t6 = sqrthalf;
        
        t1 = a[5].re;
        t2 = a[5].im - t1;
        t2 *= t6;
        t1 += a[5].im;
        t1 *= t6;
        t4 = a[7].im;
        t3 = a[7].re - t4;
        t3 *= t6;
        t4 += a[7].re;
        t4 *= t6;
        
        t8 = t3 - t1;
        t1 += t3;
        t7 = t2 - t4;
        t2 += t4;
        
        t4 = a[3].im - t8;
        a[7].im = t4;
        t5 = a[1].re - t1;
        a[5].re = t5;
        t3 = a[3].re - t7;
        a[7].re = t3;
        t6 = a[1].im - t2;
        a[5].im = t6;
        
        t8 += a[3].im;
        a[3].im = t8;
        t1 += a[1].re;
        a[1].re = t1;
        t7 += a[3].re;
        a[3].re = t7;
        t2 += a[1].im;
        a[1].im = t2;
    }
    
    static void u16(cmplxT<T> *a)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        
        u8(a);
        u4(a + 8);
        u4(a + 12);
        
        UNTRANSFORMZERO(a[0],a[4],a[8],a[12]);
        UNTRANSFORMHALF(a[2],a[6],a[10],a[14]);
        UNTRANSFORM(a[1],a[5],a[9],a[13],d16[0].re,d16[0].im);
        UNTRANSFORM(a[3],a[7],a[11],a[15],d16[0].im,d16[0].re);
    }
    
    /* a[0...8n-1], w[0...2n-2] */
    static void upass(cmplxT<T> *a,const cmplxT<T> *w,uint32_t n)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        cmplxT<T> *a1;
        cmplxT<T> *a2;
        cmplxT<T> *a3;
        
        a2 = a + 4 * n;
        a1 = a + 2 * n;
        a3 = a2 + 2 * n;
        n -= 1;
        
        UNTRANSFORMZERO(a[0],a1[0],a2[0],a3[0]);
        UNTRANSFORM(a[1],a1[1],a2[1],a3[1],w[0].re,w[0].im);
        
        for (;;) {
            UNTRANSFORM(a[2],a1[2],a2[2],a3[2],w[1].re,w[1].im);
            UNTRANSFORM(a[3],a1[3],a2[3],a3[3],w[2].re,w[2].im);
            if (!--n) break;
            a += 2;
            a1 += 2;
            a2 += 2;
            a3 += 2;
            w += 2;
        }
    }
    
    static void u32(cmplxT<T> *a)
    {
        u16(a);
        u8(a + 16);
        u8(a + 24);
        upass(a,d32,4);
    }
    
    static void u64(cmplxT<T> *a)
    {
        u32(a);
        u16(a + 32);
        u16(a + 48);
        upass(a,d64,8);
    }
    
    static void u128(cmplxT<T> *a)
    {
        u64(a);
        u32(a + 64);
        u32(a + 96);
        upass(a,d128,16);
    }
    
    static void u256(cmplxT<T> *a)
    {
        u128(a);
        u64(a + 128);
        u64(a + 192);
        upass(a,d256,32);
    }
    
    static void u512(cmplxT<T> *a)
    {
        u256(a);
        u128(a + 256);
        u128(a + 384);
        upass(a,d512,64);
    }
    
    
    /* a[0...8n-1], w[0...n-2]; n even, n >= 4 */
    static void upassbig(cmplxT<T> *a,const cmplxT<T> *w,uint32_t n)
    {
        T t1, t2, t3, t4, t5, t6, t7, t8;
        cmplxT<T> *a1;
        cmplxT<T> *a2;
        cmplxT<T> *a3;
        uint32_t k;
        
        a2 = a + 4 * n;
        a1 = a + 2 * n;
        a3 = a2 + 2 * n;
        k = n - 2;
        
        UNTRANSFORMZERO(a[0],a1[0],a2[0],a3[0]);
        UNTRANSFORM(a[1],a1[1],a2[1],a3[1],w[0].re,w[0].im);
        a += 2;
        a1 += 2;
        a2 += 2;
        a3 += 2;
        
        do {
            UNTRANSFORM(a[0],a1[0],a2[0],a3[0],w[1].re,w[1].im);
            UNTRANSFORM(a[1],a1[1],a2[1],a3[1],w[2].re,w[2].im);
            a += 2;
            a1 += 2;
            a2 += 2;
            a3 += 2;
            w += 2;
        } while (k -= 2);
        
        UNTRANSFORMHALF(a[0],a1[0],a2[0],a3[0]);
        UNTRANSFORM(a[1],a1[1],a2[1],a3[1],w[0].im,w[0].re);
        a += 2;
        a1 += 2;
        a2 += 2;
        a3 += 2;
        
        k = n - 2;
        do {
            UNTRANSFORM(a[0],a1[0],a2[0],a3[0],w[-1].im,w[-1].re);
            UNTRANSFORM(a[1],a1[1],a2[1],a3[1],w[-2].im,w[-2].re);
            a += 2;
            a1 += 2;
            a2 += 2;
            a3 += 2;
            w -= 2;
        } while (k -= 2);
    }
    
    
    
    static void u1024(cmplxT<T> *a)
    {
        u512(a);
        u256(a + 512);
        u256(a + 768);
        upassbig(a,d1024,128);
    }
    
    static void u2048(cmplxT<T> *a)
    {
        u1024(a);
        u512(a + 1024);
        u512(a + 1536);
        upassbig(a,d2048,256);
    }
    
    
    static void u4096(cmplxT<T> *a)
    {
        u2048(a);
        u1024(a + 2048);
        u1024(a + 3072);
        upassbig(a,d4096,512);
    }
    
    static void u8192(cmplxT<T> *a)
    {
        u4096(a);
        u2048(a + 4096);
        u2048(a + 6144);
        upassbig(a,d8192,1024);
    }
    
    static void u16384(cmplxT<T> *a)
    {
        u8192(a);
        u4096(a + 8192);
        u4096(a + 8192 + 4096);
        upassbig(a,d16384,2048);
    }
    
    static void u32768(cmplxT<T> *a)
    {
        u16384(a);
        u8192(a + 16384);
        u8192(a + 16384  + 8192 );
        upassbig(a,d32768,4096);
    }
    
    
    static void __fft_gen(cmplxT<T> *buf, const cmplxT<T> *buf2, int32_t sz, int32_t isfull)
    {
        int32_t x;
        T div=M_PI*0.25/(sz+1);
        
        if (isfull) div*=2.0;
        
        for (x = 0; x < sz; x ++)
        {
            if (!(x & 1) || !buf2)
            {
                buf[x].re = (T) F_COS((T)(x+1)*div);
                buf[x].im = (T) F_SIN((T)(x+1)*div);
            } else
            {
                buf[x].re = buf2[x >> 1].re;
                buf[x].im = buf2[x >> 1].im;
            }
        }
    }
    
#ifndef WDL_FFT_NO_PERMUTE
    
    static uint32_t fftfreq_c(uint32_t i,uint32_t n)
    {
        uint32_t m;
        
        if (n <= 2) return i;
        
        m = n >> 1;
        if (i < m) return fftfreq_c(i,m) << 1;
        
        i -= m;
        m >>= 1;
        if (i < m) return (fftfreq_c(i,m) << 2) + 1;
        i -= m;
        return ((fftfreq_c(i,m) << 2) - 1) & (n - 1);
    }
    
    static void idx_perm_calc(int32_t offs, int32_t n)
    {
        int32_t i, j;
        _idxperm[offs] = 0;
        for (i = 1; i < n; ++i) {
            j = fftfreq_c(i, n);
            _idxperm[offs+n-j] = i;
        }
    }
    
    static __inline int32_t WDL_fft_permute(int32_t fftsize, int32_t idx)
    {
        return _idxperm[fftsize + idx - 2];
    }
    
    static __inline int32_t *WDL_fft_permute_tab(int32_t fftsize)
    {
        return &_idxperm[fftsize - 2];
    }
    
    
#endif
    
    
    /* Expects cmplxT<T> input[0..len-1] scaled by 1.0/len, returns
    cmplxT<T> output[0..len-1] order by WDL_fft_permute(len). */

    static void fft(cmplxT<T> *buf, int32_t len, int32_t isInverse)
    {
        switch (len)
        {
            case 2: c2(buf); break;
#define TMP(x) case x: if (!isInverse) c##x(buf); else u##x(buf); break;
                TMP(4)
                TMP(8)
                TMP(16)
                TMP(32)
                TMP(64)
                TMP(128)
                TMP(256)
                TMP(512)
                TMP(1024)
                TMP(2048)
                TMP(4096)
                TMP(8192)
                TMP(16384)
                TMP(32768)
#undef TMP
        }
    }
    
    static inline void r2(T *a)
    {
        T t1, t2;
        
        t1 = a[0] + a[1];
        t2 = a[0] - a[1];
        a[0] = t1 * 2;
        a[1] = t2 * 2;
    }
    
    static inline void v2(T *a)
    {
        T t1, t2;
        
        t1 = a[0] + a[1];
        t2 = a[0] - a[1];
        a[0] = t1;
        a[1] = t2;
    }
    
    static void two_for_one(T* buf, const cmplxT<T> *d, int32_t len, int32_t isInverse)
    {
        const uint32_t half = (uint32_t)len >> 1, quart = half >> 1, eighth = quart >> 1;
        const int32_t *permute = WDL_fft_permute_tab(half);
        uint32_t i, j;
        
        cmplxT<T> *p, *q, tw, sum, diff;
        T tw1, tw2;
        
        if (!isInverse)
        {
            fft((cmplxT<T>*)buf, half, isInverse);
            r2(buf);
        } else
        {
            v2(buf);
        }
        
        /* Source: http://www.katjaas.nl/realFFT/realFFT2.html */
        
        for (i = 1; i < quart; ++i)
        {
            p = (cmplxT<T>*)buf + permute[i];
            q = (cmplxT<T>*)buf + permute[half - i];
            
            /*  tw.re = cos(2*PI * i / len);
             tw.im = sin(2*PI * i / len); */
            
            if (i < eighth)
            {
                j = i - 1;
                tw = d[j];
            } else if (i > eighth)
            {
                j = quart - i - 1;
                tw = d[j];
            } else
            {
                tw.re = tw.im = sqrthalf;
            }
            
            if (!isInverse) tw.re = -tw.re;
            
            sum.re  = p->re + q->re;
            sum.im  = p->im + q->im;
            diff.re = p->re - q->re;
            diff.im = p->im - q->im;
            
            tw1 = tw.re * sum.im + tw.im * diff.re;
            tw2 = tw.im * sum.im - tw.re * diff.re;
            
            p->re = sum.re - tw1;
            p->im = diff.im - tw2;
            q->re = sum.re + tw1;
            q->im = -(diff.im + tw2);
        }
        
        p = &((cmplxT<T>*)buf)[permute[i]];
        p->re *=  2;
        p->im *= -2;
        
        if (isInverse) fft((cmplxT<T>*)buf, half, isInverse);
    }
    
    static int32_t *fft_reorder_table_for_size(int32_t fftsize)
    {
        int bitsz = floorlog2(fftsize);
        if (bitsz <= FFT_MINBITLEN_REORDER)
            return s_tab;
        return s_tab + (1 << bitsz) + (bitsz - FFT_MINBITLEN_REORDER) * 24;
    }
    
    static int32_t *fft_reorder_table_for_bitsize(int32_t bitsz)
    {
        if (bitsz <= FFT_MINBITLEN_REORDER)
            return s_tab;
        return s_tab + (1 << bitsz) + (bitsz - FFT_MINBITLEN_REORDER) * 24;
    }
    
    static void fft_make_reorder_table(int32_t bitsz, int32_t *tab)
    {
        const int32_t fft_sz = 1 << bitsz;
        uint8_t flag[1 << FFT_MAXBITLEN];
        int32_t x;
        memset(flag, 0, fft_sz);
        
        for (x = 0; x < fft_sz; x++)
        {
            int32_t fx = 0;
            if (!flag[x] && (fx = WDL_fft_permute(fft_sz, x)) != x)
            {
                flag[x] = 1;
                *tab++ = x;
                do
                {
                    flag[fx] = 1;
                    *tab++ = fx;
                    fx = WDL_fft_permute(fft_sz, fx);
                } while (fx != x);
                *tab++ = 0; // delimit a run
            }
            else flag[x] = 1;
        }
        *tab++ = 0; // doublenull terminated
    }
    
    static void WDL_fft_init()
    {
        static bool ffttabinit = false;
        
        if (!ffttabinit)
        {
            int32_t i, offs;
            ffttabinit=true;
            
            fprintf(stderr, "WDL_fft_init()\n");
            
#define fft_gen(x,y,z) __fft_gen(x,y,sizeof(x)/sizeof(x[0]),z)
            fft_gen(d16,0,1);
            fft_gen(d32,d16,1);
            fft_gen(d64,d32,1);
            fft_gen(d128,d64,1);
            fft_gen(d256,d128,1);
            fft_gen(d512,d256,1);
            fft_gen(d1024,d512,0);
            fft_gen(d2048,d1024,0);
            fft_gen(d4096,d2048,0);
            fft_gen(d8192,d4096,0);
            fft_gen(d16384,d8192,0);
            fft_gen(d32768,d16384,0);
#undef fft_gen
            
#ifndef WDL_FFT_NO_PERMUTE
            offs = 0;
            for (i = 2; i <= 32768; i *= 2)
            {
                idx_perm_calc(offs, i);
                offs += i;
            }
#endif
        }
    }
    
};



#define DECL_WDLFFT(TYPE) \
template <typename T> int32_t WDLFFT<T>::s_tab[WDLFFT<T>::S_TAB_SIZE]; \
template <typename T> int32_t WDLFFT<T>::_idxperm[WDLFFT<T>::IDXPERM_SIZE]; \
template <typename T> cmplxT<T> WDLFFT<T>::d16[3]; \
template <typename T> cmplxT<T> WDLFFT<T>::d32[7]; \
template <typename T> cmplxT<T> WDLFFT<T>::d64[15]; \
template <typename T> cmplxT<T> WDLFFT<T>::d128[31]; \
template <typename T> cmplxT<T> WDLFFT<T>::d256[63]; \
template <typename T> cmplxT<T> WDLFFT<T>::d512[127]; \
template <typename T> cmplxT<T> WDLFFT<T>::d1024[127]; \
template <typename T> cmplxT<T> WDLFFT<T>::d2048[255]; \
template <typename T> cmplxT<T> WDLFFT<T>::d4096[511]; \
template <typename T> cmplxT<T> WDLFFT<T>::d8192[1023]; \
template <typename T> cmplxT<T> WDLFFT<T>::d16384[2047]; \
template <typename T> cmplxT<T> WDLFFT<T>::d32768[4095];



