#include "stdio.h"
#include "math.h"

#define MACHINE_EPSILON(type)                       \
                                                    \
int get_epsilon_##type(){                           \
    type epsilon = 1;                               \
    size_t counter = 0;                             \
    while(1 + epsilon / 2 > 1){                     \
        epsilon /= 2;                               \
        counter++;                                  \
    }                                               \
    return counter;                                 \
}                                                   \
                                                    \
int get_inf_##type(){                               \
    type tmp = 1;                                   \
    size_t counter = -1;                            \
    while(tmp * 2 > tmp){                           \
        tmp *= 2;                                   \
        counter++;                                  \
    }                                               \
    return counter;                                 \
}                                                   \
                                                    \
int get_zero_##type(){                              \
    type tmp = 1;                                   \
    size_t counter = 0;                             \
    while(tmp / 2 < tmp){                           \
        tmp /= 2;                                   \
        counter++;                                  \
    }                                               \
    return counter;                                 \
}                                                   \
                                                    \
void comparator_##type(){                           \
    type epsilon = 1;                               \
    while(1 + epsilon / 2 > 1){                     \
        epsilon /= 2;                               \
    }                                               \
    printf("%d: 1 == e/2 + 1\n", 1 == (epsilon/2 + 1));\
    printf("%d: 1 + e > e/2 + 1\n", (1 + epsilon) > (epsilon/2 + 1));\
    printf("%d: 1 + e < 1 + e + e/2\n", (1 + epsilon) < (1 + epsilon + epsilon/2));\
    printf("%d: 1 + e/2 + e < 1 + e + e/2\n", (1 + epsilon/2 + epsilon) < (1 + epsilon + epsilon/2));\
}


MACHINE_EPSILON(float);
MACHINE_EPSILON(double);


int main(){

    // x_norm = (-1)^s * (1 + M/2^n) * 2^(E + 1 - 2^(w-1))
    // x_subnorm = (-1)^s * (0 + M/2^n) * 2^(2 - 2^(w-1))

    // for float
    // 1 <= E <= 2^w - 2 - exp 
    // E = 8 bit
    // M - 23 bit
    // bias = 1-2^(w-1) = 127  - 0x01111111
    
    // for double
    // E = 11 bit
    // M = 52 bit 
    // bias = 1023 bit
    
    printf("%u epsilon for float\n", get_epsilon_float());
    printf("%u epsilon for double\n\n", get_epsilon_double());

    printf("%u %f inf for float\n", get_inf_float(), ceil(log2(get_inf_float()) + 1));
    printf("%u %f inf for double\n\n", get_inf_double(), ceil(log2(get_inf_double()) + 1));
    // 0x 7f7f ffff (max norm for f) = 2 ^ 127
    // 0x 7fef ffff ffff ffff (max norm for d) =  (1 + (1 – 2^–52))×2^1023


    // printf("%u %f zero for float\n", get_zero_float(), ceil(log2(get_zero_float()) + 1));
    // printf("%u %f zero for double\n\n", get_zero_double(), ceil(log2(get_zero_double()) + 1));

    printf("%u zero for float\n", get_zero_float());
    printf("%u zero for double\n\n", get_zero_double());
    // 0x 0000 0001 ( min subnorm for f) = 2 ^ -149
    // 0x 0000 0000 0000 0001 (min subnorm for d) = 2 ^ (–1022–52) 

    comparator_float();
    return 0;
}