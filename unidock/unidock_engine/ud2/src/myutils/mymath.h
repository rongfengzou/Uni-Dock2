//
// Created by Congcong Liu on 24-9-30.
//

#ifndef MATH_H
#define MATH_H

#include "myutils/common.h"
#include <math.h>
#include <cassert>


SCOPE_INLINE Real map_01_to_dot5(Real a){
    assert(0 < a && a <= 1);
    return a > 0.5? a: a - 1;
}

/**
 * @brief Normalize an angle to the range [-π, π)
 * @snippet test_math.cpp normalize_angle
 * @param x The angle, in radiant
 * @return The normalized angle
 */
SCOPE_INLINE Real normalize_angle(Real x) {
    x = fmod(x + PI, 2 * PI);  // Normalize to the range [0, 2π)
    if (x < 0) {
        x += 2 * PI;  // If the result is negative, normalize to [0, 2π)
    }
    x -= PI;  // Normalized to the range [-π, π)
    return x;
}

/**
 * @brief Transform an angle in [-180, 180] to radian in [-π, π]
 * @snippet test_math.cpp ang_to_rad
 * @param angle The angle in degree
 * @return The angle in radian
 */
SCOPE_INLINE Real ang_to_rad(Real angle) {
    assert(angle >= -180 && angle <= 180);
    return angle * DEG2RAD;
}

/**
 * @brief Transform an radian angle in [-π, π] to degree in [-180, 180]
 * @snippet test_math.cpp rad_to_ang
 * @param rad The angle in radian
 * @return The angle in degree
 */
SCOPE_INLINE Real rad_to_ang(Real rad) {
    assert(rad >= -PI && rad <= PI);
    return rad * RAD2DEG;
}


//------------------------------------------------------
//------------------- Range Operations -----------------
//------------------------------------------------------

/**
 * @brief Check if a value is in a given range
 * @snippet test/unit/myutils/test_math.cpp isInRanges
 * @param value The value to check
 * @param ranges The range array, each range is represented by two values, indicating the start and end of the range
 * @param num_range The number of ranges
 * @return True if the value is in any range, false otherwise
 */
SCOPE_INLINE bool isInRanges(Real value, int* ranges, int num_range) {
    for (int i = 0; i < num_range; i++) {
        if (value >= ranges[i * 2] && value <= ranges[i * 2 + 1]) {
            return true;
        }
    }
    return false;
}


/**
 * @brief Randomly select a value from a given range fixme: the boundary values can be missed
 * @snippet test/unit/myutils/test_math.cpp get_radian_in_ranges
 * @param ranges The range array, each range is represented by two values, indicating the start and end of the range
 * @param num_range The number of ranges
 * @param rand_2 2 random numbers, both in the range (0, 1]
 * @return The randomly selected value in the given range
 */
SCOPE_INLINE Real get_radian_in_ranges(Real* ranges, int num_range, Real *rand_2){
    int i_range = ceil(rand_2[0] * num_range - 1); //(0, 1] mapped to [0, 1, ..., num_range - 1]

    // choose a value:
    Real from = ranges[i_range * 2];
    Real to = ranges[i_range * 2 + 1];
    return from + (to - from) * rand_2[1]; // (0, 1] mapped to [from, ..., to]
}

SCOPE_INLINE Real clamp_by_range(Real ang, Real hi, Real lo ){
    if (ang > hi){
        return hi;
    }
    if (ang < lo){
        return lo;
    }
    return ang;
}



SCOPE_INLINE Real clamp_by_ranges(Real ang, Real* ranges, int nrange) {
    // return a1 if a1 in any range
    for (int i = 0; i < nrange; i++) {
        if (ang >= ranges[2 * i] && ang <= ranges[2 * i + 1]) {
            return ang;
        }
    }

    // return the closest boundary value if ang is not in any range
    Real closest_ang = ang;
    Real min_distance = 100.;
    for (int i = 0; i < nrange; i++) {
        Real distance_to_min = fabs(ang - ranges[2 * i]);
        Real distance_to_max = fabs(ang - ranges[2 * i + 1]);

        if (distance_to_min < min_distance) {
            min_distance = distance_to_min;
            closest_ang = ranges[2 * i];
        }
        if (distance_to_max < min_distance) {
            min_distance = distance_to_max;
            closest_ang = ranges[2 * i + 1];
        }
    }
    return closest_ang;
}

/**
 * @brief Generate a real number in [min, max]
 *
 * @param c A random integer within Integer range
 * @param min Minimum value (Real)
 * @param max Maximum value (Real)
 * @param n Number of divisions. The larger n is, the more accurate the result is.
 * @return A real number in [min, max]
 */
SCOPE_INLINE Real get_real_within_by_int(uint c, Real min, Real max, int n=31) {//todo: automatically adjust n
    Real tmp = c % n;
    Real v = tmp * (max - min) / n + min;
    if (v < min){
        return min;
    }
    if(v > max){
        return max;
    }
    return v;
}


//------------------------------------------------------
//------------------- Math functions -------------------
//------------------------------------------------------

/**
 * @brief Compute the value and derivative of a Gaussian function
 * @snippet test/unit/myutils/test_math.cpp gaussian
 * @param d Distance
 * @param offset Offset of the Gaussian function
 * @param width Width of the Gaussian function
 * @param out_f Output derivative
 * @return Value of the Gaussian function
 */
SCOPE_INLINE Real gaussian(Real d, Real offset, Real width, Real *out_f){
    Real e = exp(-square((d - offset) / width));
    *out_f = e * 2 * (offset - d) / width / width;
    return e;
}


//------------------------------------------------------
//------------------- Vector Operations ----------------
//------------------------------------------------------

/**
 * @brief Compute the cross product of two 3D vectors
 * @snippet test/unit/myutils/test_math.cpp cross_product
 * @param a The first vector
 * @param b The second vector
 * @param out_res The result vector
 */
template <typename T1, typename T2, typename T3>
SCOPE_INLINE void cross_product(const T1 *a, const T2 *b, T3 *out_res) {
    out_res[0] = a[1] * b[2] - a[2] * b[1];
    out_res[1] = a[2] * b[0] - a[0] * b[2];
    out_res[2] = a[0] * b[1] - a[1] * b[0];
}


/**
 * @brief Compute the dot product of two 3D vectors
 * @snippet test/unit/myutils/test_math.cpp dot_product
 * @param a The first vector
 * @param b The second vector
 * @return The dot product of the two vectors
 */
template <typename T1, typename T2>
SCOPE_INLINE Real dot_product(const T1 *a, const T2 *b) {
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}



/**
 * @brief Calculate the norm of a vector
 * @snippet test_math.cpp norm_vec3
 * @param a The vector
 * @return The norm of the vector
 */
SCOPE_INLINE Real norm_vec3(const Real *a) {
    return sqrt(a[0] * a[0] + a[1] * a[1] + a[2] * a[2]);
}

/**
 * @brief Calculate the squared distance between two vectors
 * @snippet test_math.cpp dist_norm2
 * @param a The first vector
 * @param b The second vector
 * @return The squared distance between the two vectors
 */
SCOPE_INLINE Real dist2(const Real *a, const Real *b) {
    return (a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1])
           + (a[2] - b[2]) * (a[2] - b[2]);
}

SCOPE_INLINE Real dist(const Real *a, const Real *b) {
    return sqrt(dist2(a, b));
}

//------------------------------------------------------
//------------------- Matrix Operations ----------------
//------------------------------------------------------

/**
 * @brief Initialize a 3x3 matrix with a given value
 * @snippet test_math.cpp init_3x3_mat
 * @param out_m The matrix to be initialized
 * @param fill_value The value to fill the matrix with
 */
SCOPE_INLINE void init_3x3_mat(Real *out_m, Real fill_value) {
    // fixed to 3x3 matrix_d
    for (int i = 0; i < 9; i++) {
        out_m[i] = fill_value;
    }
}

    
/**
 * @brief Set an element of a matrix
 * @snippet test_math.cpp mat_set_element
 * @param out_m The matrix to be set
 * @param dim The dimension of the matrix
 * @param i The row index
 * @param j The column index
 * @param fill_value The value to set the element to
 */
SCOPE_INLINE void mat_set_element(Real *out_m, int dim, int i, int j, float fill_value) {
    out_m[i + j * dim] = fill_value;
}


/**
 * @brief Calculate the sequential index for an upper triangular matrix (i <= j)
 * @snippet test_math.cpp uptri_mat_index
 * @param i The row index
 * @param j The column index
 * @return The index of the element in the sequential triangular matrix
 */
SCOPE_INLINE int uptri_mat_index(int i, int j) {
    // assert(i <= j);
    // from i_col, i_row to ind in sequence
    return i + j * (j + 1) / 2;
}

/**
 * @brief Calculate the sequential index for a triangular matrix, including upper- & lower-
 * @snippet test_math.cpp tri_mat_index
 * @param i The row index
 * @param j The column index
 * @return The index of the element in the sequential triangular matrix
 */
SCOPE_INLINE int tri_mat_index(int i, int j) {
    return (i < j) ? uptri_mat_index(i, j) : uptri_mat_index(j, i);
}


#endif //MATH_H
