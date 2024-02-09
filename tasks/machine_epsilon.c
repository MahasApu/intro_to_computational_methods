#include "stdio.h"
#include "math.h"

int get_epsilon(){
    float epsilon = 1;
    size_t counter = 0;
    while(1 + epsilon / 2 > 1){
        epsilon /= 2;
        counter++;
    }
    return counter;
}

int main(){
    printf("%u", get_epsilon());
}