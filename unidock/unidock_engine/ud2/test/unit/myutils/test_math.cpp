//
// Created by Congcong Liu on 24-10-16.
//

#include <catch2/catch_amalgamated.hpp>
#include "myutils/mymath.h"

TEST_CASE("normalize_angle normalizes an angle to the range [-π, π)", "[normalize_angle]") {
    Real margin = 1e-4;
    Real res = normalize_angle(5316.74512);
    res = normalize_angle(5319.620117);
    REQUIRE_THAT(normalize_angle(0.0f), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(PI), Catch::Matchers::WithinAbs(-PI, margin)); // 注意，PI 会被转换为 -PI
    REQUIRE_THAT(normalize_angle(-PI), Catch::Matchers::WithinAbs(-PI, margin));
    REQUIRE_THAT(normalize_angle(2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(-2 * PI), Catch::Matchers::WithinAbs(0.0f, margin));
    REQUIRE_THAT(normalize_angle(3 * PI), Catch::Matchers::WithinAbs(-PI, margin));
    REQUIRE_THAT(normalize_angle(-3 * PI), Catch::Matchers::WithinAbs(-PI, margin));
    REQUIRE_THAT(normalize_angle(0.3), Catch::Matchers::WithinAbs(0.3, margin));
    REQUIRE_THAT(normalize_angle(0.3 + 2 * PI), Catch::Matchers::WithinAbs(0.3, margin));
    REQUIRE_THAT(normalize_angle(-0.13 + 2 * PI), Catch::Matchers::WithinAbs(-0.13, margin));

    //![normalize_angle]
}

TEST_CASE("ang_to_rad converts an angle in degree to radian", "[ang_to_rad]") {
    REQUIRE_THAT(ang_to_rad(0), Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(ang_to_rad(180), Catch::Matchers::WithinAbs(PI, 1e-4));
    REQUIRE_THAT(ang_to_rad(90), Catch::Matchers::WithinAbs(PI / 2, 1e-4));
    REQUIRE_THAT(ang_to_rad(-180), Catch::Matchers::WithinAbs(-PI, 1e-4));
    //![ang_to_rad]
}


TEST_CASE("get_radian_in_ranges randomly selects a value within a given range", "[get_radian_in_ranges]") {
    Real ranges[] = {-180, -160, 20, 90, 110, 150};
    int num_range = 3;

    Real rand[] = {0.54f, 0.1f};
    REQUIRE(get_radian_in_ranges(ranges, num_range, rand) == 27);

    Real rand2[] = {0.89f, 0.0001f};
    REQUIRE(get_radian_in_ranges(ranges, num_range, rand2) == 110);

    Real rand3[] = {0.21f, 1.0f}  ;
    REQUIRE(get_radian_in_ranges(ranges, num_range, rand3) == -160);

    Real rand4[] = {0.0001f, 0.00002f}; //won't be zero
    REQUIRE(get_radian_in_ranges(ranges, num_range, rand4) == -179);

    Real rand5[] = {1.0f, 0.999f}; //fixme: won't be zero
    REQUIRE(get_radian_in_ranges(ranges, num_range, rand5) == 149);
    //![get_radian_in_ranges]
}


TEST_CASE("get_real_within_by_int", "[get_real_within_by_int]") {
    // default n = 1001
    REQUIRE_THAT(get_real_within_by_int(0, 0, 10), Catch::Matchers::WithinAbs(0, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(11, 0, 10), Catch::Matchers::WithinAbs(11.0/1001 * 10, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(1002, 0, 10), Catch::Matchers::WithinAbs(1.0/1001 * 10, 1e-4));
    REQUIRE_THAT(get_real_within_by_int(10010000, 0, 10), Catch::Matchers::WithinAbs(0, 1e-4));
    // x
    REQUIRE_THAT(get_real_within_by_int(633093634, -30.5, -5.5), Catch::Matchers::WithinAbs(-26.179321, 1e-4));
    // y
    REQUIRE_THAT(get_real_within_by_int(2151943577, 2.6999998, 27.7), Catch::Matchers::WithinAbs(22.280418, 1e-4));
    // pi pi
    REQUIRE_THAT(get_real_within_by_int(633, -PI, PI), Catch::Matchers::WithinAbs(0.831690311f, 1e-4));

    //![get_real_within_by_int]
}


TEST_CASE("compute gaussian", "[gaussian]") {
    Real e, f;
    e = gaussian(0, 0, 3, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = gaussian(1, 1, 3, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(1, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0, 1e-4));

    e = gaussian(0, 1, 4, &f);
    REQUIRE_THAT(e, Catch::Matchers::WithinAbs(0.939413071f, 1e-4));
    REQUIRE_THAT(f, Catch::Matchers::WithinAbs(0.117426634f, 1e-4));
}


TEST_CASE("computes the cross product of two 3D vectors", "[cross_product]") {
    Real vec1[] = {1.0f, 2.0f, 3.0f};
    Real vec2[] = {4.0f, 5.0f, 6.0f};
    Real result[3];
    cross_product(vec1, vec2, result);
    REQUIRE_THAT(result[0], Catch::Matchers::WithinAbs(-3.0f, 1e-4));
    REQUIRE_THAT(result[1], Catch::Matchers::WithinAbs(6.0f, 1e-4));
    REQUIRE_THAT(result[2], Catch::Matchers::WithinAbs(-3.0f, 1e-4));


    double vec3[] = {1.0, 0.0, 0.0};
    double vec4[] = {0.0, 1.0, 0.0};
    double result2[3];
    cross_product(vec3, vec4, result2);
    REQUIRE_THAT(result2[0], Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(result2[1], Catch::Matchers::WithinAbs(0.0, 1e-4));
    REQUIRE_THAT(result2[2], Catch::Matchers::WithinAbs(1.0, 1e-4));

    int vec5[] = {1, 0, 0};
    int vec6[] = {0, 1, 0};
    int result3[3];
    cross_product(vec5, vec6, result3);
    REQUIRE(result3[0] == 0);
    REQUIRE(result3[1] == 0);
    REQUIRE(result3[2] == 1);   
    //![cross_product]
}   

TEST_CASE("compute dot product", "[dot_product]") {
    Real vec1[] = {1.0f, 2.0f, 3.0f};
    Real vec2[] = {4.0f, 5.0f, 6.0f};
    REQUIRE_THAT(dot_product(vec1, vec2), Catch::Matchers::WithinAbs(32.0f, 1e-4));
    //![dot_product]
}

TEST_CASE("norm_vec3 computes the Euclidean norm of a 3D vector", "[norm_vec3]") {
    Real margin = 1e-4;

    Real vec1[] = {3.0f, 4.0f, 0.0f};
    REQUIRE_THAT(norm_vec3(vec1), Catch::Matchers::WithinAbs(5.0f, margin));

    Real vec2[] = {1.0f, 2.0f, 2.0f};
    REQUIRE_THAT(norm_vec3(vec2), Catch::Matchers::WithinAbs(3.0f, margin));

    Real vec3[] = {0.0f, 0.0f, 0.0f};
    REQUIRE_THAT(norm_vec3(vec3), Catch::Matchers::WithinAbs(0.0f, margin));
    //![norm_vec3]
}

TEST_CASE("dist2 computes the squared Euclidean distance between two 3D points", "[dist2]") {
    Real margin = 1e-4;

    Real point1[] = {1.0f, 2.0f, 3.0f};
    Real point2[] = {4.0f, 6.0f, 8.0f};
    REQUIRE_THAT(dist2(point1, point2), Catch::Matchers::WithinAbs(50.0f, margin));

    Real point3[] = {0.0f, 0.0f, 0.0f};
    Real point4[] = {0.0f, 0.0f, 0.0f};
    REQUIRE_THAT(dist2(point3, point4), Catch::Matchers::WithinAbs(0.0f, margin));

    Real point5[] = {1.0f, 1.0f, 1.0f};
    Real point6[] = {2.0f, 2.0f, 2.0f};
    REQUIRE_THAT(dist2(point5, point6), Catch::Matchers::WithinAbs(3.0f, margin));
    //![dist2]
}


TEST_CASE("init_3x3_mat initializes a 3x3 matrix with a given value", "[init_3x3_mat]") {
    Real m[9];
    init_3x3_mat(m, 1.0f);
    for (int i = 0; i < 9; ++i) {
        REQUIRE_THAT(m[i], Catch::Matchers::WithinAbs(1.0f, 1e-4));
    }
    //![init_3x3_mat]
}

TEST_CASE("mat_set_element sets an element of a 3x3 matrix", "[mat_set_element]") {
    Real m[9];
    mat_set_element(m, 3, 1, 2, 3.0f);
    REQUIRE_THAT(m[1 + 2 * 3], Catch::Matchers::WithinAbs(3.0f, 1e-4));
    //![mat_set_element]
}

TEST_CASE("uptri_mat_index calculates the sequential index for a triangular matrix", "[uptri_mat_index]") {
    REQUIRE(uptri_mat_index(0, 0) == 0);
    REQUIRE(uptri_mat_index(1, 1) == 2);
    REQUIRE(uptri_mat_index(0, 1) == 1);
    //![uptri_mat_index]
}   

TEST_CASE("tri_mat_index calculates the sequential index for a triangular matrix, allowing i <= j or i >= j", "[tri_mat_index]") {
    REQUIRE(tri_mat_index(0, 0) == 0);
    REQUIRE(tri_mat_index(1, 0) == 1);
    REQUIRE(tri_mat_index(1, 1) == 2);
    REQUIRE(tri_mat_index(0, 1) == 1);
    REQUIRE(tri_mat_index(2, 0) == 3);
    REQUIRE(tri_mat_index(2, 1) == 4);
    REQUIRE(tri_mat_index(2, 2) == 5);
    //![tri_mat_index]
}   

