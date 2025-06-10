//
// Created by Congcong Liu on 24-10-17.
//

#include <catch2/catch_amalgamated.hpp>
#include "geometry/quaternion.h"


TEST_CASE("quaternion_multiply_left multiplies two quaternions", "[quaternion_multiply_left]"){
    // around Z-axis, 10 + 20 = 30 degrees
    Real q3[] = {0.9961947, 0, 0, 0.0871557};
    Real q4[] = {0.9848078, 0, 0, 0.1736482,};
    Real expected3[] = {0.9659258, 0, 0, 0.258819};

    quaternion_multiply_left(q4, q3);
    for (int i = 0; i < 4; ++i){
        REQUIRE_THAT(q3[i], Catch::Matchers::WithinAbs(expected3[i], 1e-4));
    }
    //! [quaternion_multiply_left]
}


TEST_CASE("normalize_quaternion normalizes a quaternion", "[normalize_quaternion]"){
    Real q1[] = {1.0f, 2.0f, 3.0f, 4.0f};
    Real expected[] = {0.1826f, 0.3651f, 0.5477f, 0.7303f};

    quaternion_normalize(q1);
    for (int i = 0; i < 4; ++i){
        REQUIRE_THAT(q1[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
    }
    //! [normalize_quaternion]
}

TEST_CASE("quaternion_increment adds two quaternions", "[quaternion_increment]"){
    Real q1[] = {0.9238795, 0, 0, 0.3826834}; // Z 45D
    Real q2[] = {0.7071068, 0.5, -0.5, 0}; // (-1, 1, 0), -90D
    Real expected[] = {0.653282, 0.270598, -0.653282, 0.270598}; // Y -90D
    quaternion_normalize(q1);
    quaternion_normalize(q2);
    quaternion_increment(q1, q2);
    for (int i = 0; i < 4; ++i){
        REQUIRE_THAT(q1[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
    }
}


TEST_CASE("rotate_vec_by_quaternion rotates a vector by a quaternion", "[rotate_vec_by_quaternion]"){
    Real vec[] = {2.0f, 0.0f, 0.0f};
    Real expected[] = {0.0f, 2.0f, 0.0f};
    Real q[] = {0.7071068f, 0.f, 0.f, 0.7071068f};
    rotate_vec_by_quaternion(vec, q);
    for (int i = 0; i < 3; ++i){
        REQUIRE_THAT(vec[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
    }

    Real vec2[] = {2.f, 0.f, 0.f};
    Real expected2[] = {1.f, 1.7320508f, 0.f};
    Real q2[] = {0.8660254, 0, 0, 0.5f};
    rotate_vec_by_quaternion(vec2, q2);
    for (int i = 0; i < 3; ++i){
        REQUIRE_THAT(vec2[i], Catch::Matchers::WithinAbs(expected2[i], 1e-4));
    }

    Real vec3[] = {1.0f, 0.0f, 0.0f};
    Real expected3[] = {-0.37625361f, 0.92650867f, -0.00385983f};
    Real q3[] = {0.51421413f, 0.21784632f, 0.32676949f, 0.76246214f};
    rotate_vec_by_quaternion(vec3, q3);
    for (int i = 0; i < 3; ++i){
        REQUIRE_THAT(vec3[i], Catch::Matchers::WithinAbs(expected3[i], 1e-4));
    }

    Real vec4[] = {1.0f, 2.3f, -9.23f};
    Real expected4[] = {-8.02033915f, -2.19741106f, -4.72529836f};
    Real q4[] = {0.51421413f, 0.21784632f, 0.32676949f, 0.76246214f};
    rotate_vec_by_quaternion(vec4, q4);
    for (int i = 0; i < 3; ++i){
        REQUIRE_THAT(vec4[i], Catch::Matchers::WithinAbs(expected4[i], 1e-4));
    }
    //! [rotate_vec_by_quaternion]
}

//! [quaternion_to_rotvec]
TEST_CASE("Convert unit quaternion to rotation vector", "[quaternion_to_rotvec]") {

    SECTION("Identity quaternion (no rotation)") {
        Real q[] = {1.0f, 0.0f, 0.0f, 0.0f};
        Real out_v[3];
        Real expected[] = {0.0f, 0.0f, 0.0f};

        quaternion_to_rotvec(out_v, q);
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(out_v[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
        }
    }

    SECTION("180-degree rotation around Z-axis") {
        Real q[] = {0.0f, 0.0f, 0.0f, 1.0f};  // w=0, (x,y,z)=(0,0,1)
        Real out_v[3] = {0.};
        Real expected[] = {0.0f, 0.0f, PI};  // (0,0,π)

        quaternion_to_rotvec(out_v, q);
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(out_v[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
        }
    }

    SECTION("90-degree rotation around X-axis") {
        const Real cos_pi_4 = 0.7071068f;
        Real q[] = {cos_pi_4, cos_pi_4, 0.0f, 0.0f};  // w=cos(π/4), x=sin(π/4)
        Real out_v[3] = {0.};
        Real expected[] = {PI / 2, 0.0f, 0.0f};  // (π/2,0,0)

        quaternion_to_rotvec(out_v, q);
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(out_v[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
        }
    }

    SECTION("Arbitrary axis rotation") {
        Real q[] = {0.9586831, 0.2370836, 0.0848071, -0.1323853};  // Normalized quaternion
        Real out_v[3] = {0.};
        Real expected[] = {0.4808074, 0.1719894, -0.2684785};

        quaternion_to_rotvec(out_v, q);
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(out_v[i], Catch::Matchers::WithinAbs(expected[i], 1e-4));
        }
    }

    SECTION("Small angle approximation") {
        // Test case where sin(theta/2) approaches zero
        Real q[] = {0.9999999f, 0.0001f, 0.0f, 0.0f};
        Real out_v[3];
        Real expected[] = {0.0002f, 0.0f, 0.0f};  // Should trigger s < EPSILON

        quaternion_to_rotvec(out_v, q);
        for (int i = 0; i < 3; ++i) {
            REQUIRE_THAT(out_v[i], Catch::Matchers::WithinAbs(expected[i], 1e-4f));
        }
    }
}
//! [quaternion_to_rotvec]





















