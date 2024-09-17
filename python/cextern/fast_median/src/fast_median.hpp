#ifndef FILTER_H
#define FILTER_H

// T: float or double.
// x, y: image dimensions, width x pixels and height y pixels.
// hx, hy: window size in x and y directions.
// blockhint: preferred block size, 0 = pick automatically.
// in: input image.
// out: output image.

extern "C"
{
    void median_filter_2d_float(int nx, int ny, int box_x, int box_y, int blockhint, const float *in, float *out);
    void median_filter_2d_double(int nx, int ny, int box_x, int box_y, int blockhint, const double *in, double *out);
    void median_filter_1d_float(int nx, int box_x, int blockhint, const float *in, float *out);
    void median_filter_1d_double(int nx, int box_x, int blockhint, const double *in, double *out);
}

template <typename T>
void median_filter_2d(int nx, int ny, int hx, int hy, int blockhint, const T *in, T *out);

// As above, for the special case y = 1, hy = 0.

template <typename T>
void median_filter_1d(int x, int hx, int blockhint, const T *in, T *out);

#endif
