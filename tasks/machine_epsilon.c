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
    size_t counter = -1;                            \
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
    printf("%d: 1 + e > 1 + e + e/2\n", (1 + epsilon) < (1 + epsilon + epsilon/2));\
    printf("%d: 1 + e/2 + e < 1 + e + e/2\n", (1 + epsilon/2 + epsilon) < (1 + epsilon + epsilon/2));\
}


MACHINE_EPSILON(float);
MACHINE_EPSILON(double);


int main(){
    printf("%u \n", get_epsilon_float());
    printf("%u \n", get_epsilon_double());

    // printf("%u \n", get_inf_float());
    // printf("%u", get_inf_double());

    // printf("%u \n", get_zero_float());
    // printf("%u", get_zero_double());

    comparator_float();
}