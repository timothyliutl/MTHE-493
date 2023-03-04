#include <stdio.h>
#include <math.h>

/*
float * calc_centroids(){

return ;
}
*/

/*
run this command each time this file is changed
cc -fPIC -shared -o cosq_class.so cosq_class.c 

*/

int squared(int input){

    for(int i=0; i<100000; i++){
        int number = i;
        printf("%d \n", number);
    }

    return input * input;
}

// create function for training/fitting the scalar quantizer

void int2bin(unsigned integer, char* binary, int n)
{  
  for (int i=0;i<n;i++)   
    binary[i] = (integer & (int)1<<(n-i-1)) ? '1' : '0';
  binary[n]='\0';
}

float calc_transition_probabilities(int input, int output, float epsilon, int bits){
    char input_bin[bits];
    char output_bin[bits];

    int2bin(input, input_bin, bits);
    int2bin(output, output_bin, bits);

    float transition_prob = 1;

    for (int i=0; i<bits; i++){
        if (input_bin[i]==output_bin[i]) {
            transition_prob*=(1-epsilon);
        }else{
            transition_prob *= epsilon;
        }
    }

    return transition_prob;

}

float expected_distortion(float centroid_array[], int array_length, int index, float point, float epsilon, int bits){
    float distortion = 0;
    
    for(int i =0; i<array_length; i++){
        float trans_prob = calc_transition_probabilities(index, i, epsilon, bits);
        distortion+= trans_prob * powf(centroid_array[i] - point, 2);
    }
    return distortion;
}



int * calc_partitions(float point_array[], int point_length, float centroid_array[], int centroid_length, float epsilon, int bits, int* return_array){
    //array that has corresponding to point array, contains centroid index value for the point

    for(int i=0; i<point_length; i++){
        int index_min_dis = 0;
        float global_min;
        for(int j=0; j<centroid_length; j++){
            if(j==0){
                global_min = expected_distortion(centroid_array, centroid_length, i, point_array[j], epsilon, bits);
                continue;
            }
            float current_exp_dist = expected_distortion(centroid_array, centroid_length, i, point_array[j], epsilon, bits);
            if (current_exp_dist<global_min){
                global_min = current_exp_dist;
                index_min_dis = j;
            }
        }
        return_array[i] = index_min_dis;
    }
    return return_array;

}

float * calc_centroids(int partition_arr[], int partition_arr_len, float centroid_arr[], int centroid_arr_len, float point_arr[], int point_arr_len, float epsilon, int bits, float* return_arr){
    float sum_array[centroid_arr_len];
    int sum_array_count[centroid_arr_len];
    printf("---------\n");

    if(partition_arr_len!=point_arr_len){
        printf("uwu i made a fucky");
    }

    for(int i = 0; i<centroid_arr_len; i++){
        sum_array[i]=0;
        sum_array_count[i]=0;
        //initialize all elements of the array to 0
    }

    for(int i = 0; i<partition_arr_len; i++){
        sum_array[partition_arr[i]]+=point_arr[i];
        sum_array_count[partition_arr[i]]++;
    }

    for(int i=0; i<centroid_arr_len; i++){
        float new_centroid_val = 0;
        for(int j=0; j<centroid_arr_len; j++){
            float trans_prob = calc_transition_probabilities(i,j, epsilon, bits);
            float average;
            if(sum_array_count[j]>0){
                average = sum_array[j]/(float)(sum_array_count[j]);
                //printf("%f\n", sum_array[j]);
                new_centroid_val += (trans_prob*average);
            }
            printf("%f,     %f \n", average, trans_prob);
            
        }
        //printf("%f\n", new_centroid_val);
        return_arr[i] = new_centroid_val;
    }
    return return_arr;
}

