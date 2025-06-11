/*                                                         *
 *          C\C++ library for CTDS simulations             *
 *                                                         *
 *         Written by Aron Vizkeleti on 2021-12-10         *
 *               last modified 2025-01-08                  *
 *                                                         *
 *      to compile use:                                    *
 *      cc -std=c99 -fPIC -shared -o cSAT.so cSAT.c        *
 *      alternatively:                                     *
 *  gcc -std=c99 -m64 -fPIC -shared -o cSAT.so -O3 cSAT.c  *
 *                                                         */
#include <math.h>

#define M_PI		3.14159265358979323846
#define M_PI_2		1.57079632679489661923

#ifndef _BOOL
    typedef unsigned char bool;
    static const bool false = 0;
    static const bool true = 1;
#endif

//Helper functions (not to be called from outside)

int flat_idx(int i, int j, int N){
    return i*N -i*(i+1)/2 + j;
}

void matmulvec(double A[], double b[], int size1, int size2, double result[]) {
    // Iterate through each row of matrix A
    for (int i = 0; i < size1; i++) {
        // Initialize the result for the current row to zero
        result[i] = 0;
        // Iterate through each column of matrix A
        for (int j = 0; j < size2; j++) {
            // Multiply each element of the row in A with the corresponding element in b
            result[i] += A[i * size2 + j] * b[j];
        }
    }
}

double k_mi(int m, int i, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int j = 0; j < number_of_variables; j++)
    {   
        if (i != j){
            productum *= (1 - c[m*number_of_variables + j] * s[j]);
        }
    }    
    return 0.125 * productum; // only for 3SAT 0.125 = 2^-3
}

double k_mij(int m, int i, int j, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int idx = 0; idx < number_of_variables; idx++)
    {   
        if (i != idx && j != idx){
            productum *= (1 - c[m*number_of_variables + idx] * s[idx]);
        }
    }    
    return 0.125 * productum; // only for 3SAT 0.125 = 2^-3
}

double K_m(int m, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int j = 0; j < number_of_variables; j++)
    {   
            productum *= (1 - c[m*number_of_variables + j] * s[j]);
    }    
    return 0.125 * productum; // only for 3SAT 0.125 = 2^-3
}

double k_mi_K_m(int m, int i, double s[], int c[], int number_of_variables){
    double km = K_m(m, s, c, number_of_variables);
    return km*km*(1 - c[m*number_of_variables + i] * s[i]);
}

double gradV_i(int i, double s[], double a[], int c[], int number_of_variables, int number_of_clauses){
    double summ = 0.0;
    for (int m = 0; m < number_of_clauses; m++)
    {
        summ += 2*a[m]*c[m*number_of_variables + i]*k_mi_K_m(m, i, s, c, number_of_variables);
    }
    return summ;
}

double gradV_i2(int i, double s[], double a[], int c[], int number_of_variables, int number_of_clauses){
    double summ = 0.0;
    for (int m = 0; m < number_of_clauses; m++)
    {
        summ += 2*a[m]*c[m*number_of_variables + i]*(1-s[i]*c[m*number_of_variables + i])*pow(k_mi(m, i, s, c, number_of_variables), 2);
    }
    return summ;
}

double K_m_squared(int m, double s[], int c[], int number_of_variables){
    double productum = 1.0;
    for (int j = 0; j < number_of_variables; j++)
    {   
            productum *= (1 - c[m*number_of_variables + j] * s[j]);
    }    
    return 0.125 * 0.125 * productum*productum;
}

double second_order_potential_inef(int i, double s[], double b[], int c[], int N, int M){
    double summ = 0.0;
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < M; n++)
        {  
            if (n >= m) {
                summ += b[m*M + n] * ( c[m*N + i] * (1-s[i]*c[m*N + i])*pow(k_mi(m, i, s, c, N), 2) + c[n*N + i] * (1-s[i]*c[n*N + i])*pow(k_mi(n, i, s, c, N), 2));
            }
        }
    }
    return summ;
}

double second_order_potential_prune(int i, double s[], double b[], int c[], int N, int M){
    double summ = 0.0;
    for (int m = 0; m < M; m++)
    {
        for (int n = 0; n < M; n++)
        {  
            if (n >= m) {
                summ += ( c[m*N + i] * (1-s[i]*c[m*N + i])*pow(k_mi(m, i, s, c, N), 2) + c[n*N + i] * (1-s[i]*c[n*N + i])*pow(k_mi(n, i, s, c, N), 2));
                if (b[m*M + n] != 0.0){
                    summ *= b[m*M + n];
                }
            }
        }
    }
    return summ;
}

bool check_row(int k, const int row[], const int solution[]){
    //for elem in row:
    for (int i = 0; i < k; i++){
        if (row[i] > 0){
            //or_var = or_var or solution[elem-1]
            if (1 == solution[row[i]-1]){ return true; }
        }
        if (row[i] < 0)
            //or_var = or_var or not solution[-elem-1]
            if (0 == solution[-row[i]-1]){ return true; }
    }
    return false;
}

//Can be called from python

void get_Q(int N, int M,
            int c[], int L, 
            int edges[], double s[], 
            double k[], double Q[]) 
{
    // Loop over edges
    for (int j = 0; j < L; j++) {
        int m = edges[j * 2];
        int n = edges[j * 2 + 1];

        // Loop over variables
        for (int i = 0; i < N; i++) {
            // Calculate Q_ij contribution
            double contribution = c[m * N + i] * k_mi(m, i, s, c, N) * k[m] +
                                  c[n * N + i] * k_mi(n, i, s, c, N) * k[n];

            // Update Q_ij
            Q[i * L + j] = contribution;
        }
    }
}

void precompute_kmi(int N, int M, int c[], double s[], double result[]){
    for (int m = 0; m < M; m++) {
        for (int i = 0; i < N; i++){
            result[m*N + i] = k_mi(m, i, s, c, N);
        }
    }
}

void precompute_km(int N, int M, int c[], double s[], double k[]){
    for (int m = 0; m < M; m++) {
        k[m] = K_m(m, s, c, N);
    }
}

void get_R(int N, int M, int c[], int L, int edges[], double s[], double k[], double R[]) {
    // Loop over edges
    for (int j = 0; j < L; j++) {
        int m = edges[j * 2];
        int n = edges[j * 2 + 1];

        // Calculate & update R_j contribution
        R[j] = k[m] * k[n];
    }
}

//To be called from python

int check_sol(int N, int M, int k, int sol[], int clauses[]){
    //incorrect_flag = False
    //for i, row in enumerate(self.clauses)
    for (int i = 0; i < M; i++)
    {
        const int* row = &clauses[k*i];
        //incorrect_flag = True
        if ( ! check_row(k, row, sol) ){ return 0; }
    }
    return 1;
}

void rhs1_old(int N, int M, int c[], double y[], double result[]){
    double *s = y;
    double *a = y + N;
    for (int i = 0; i < N; i++)
    {
        result[i] = gradV_i(i, s, a, c, N, M);
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = y[i] * K_m(i-N, s, c, N);
    }
}

void rhs1(int N, int M, int c[], double y[], double result[]){
    //Most basic
    double *s = y;
    double *a = y + N;
    for (int i = 0; i < N; i++)
    {
        double grad_i = 0.0;
        for (int m = 0; m < M; m++) {
            grad_i += 2*a[m]*c[m*N + i]*(1-c[m*N + i]*s[i])*k_mi(m, i, s, c, N)*k_mi(m, i, s, c, N);
        }
        result[i] = grad_i;
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = y[i] * K_m(i-N, s, c, N);
    }
}

void rhs2(int N, int M, int c[], double y[], double result[]){
    //Aux variables updated with K_m squared
    double *s = y;
    double *a = y + N;
    for (int i = 0; i < N; i++)
    {
        result[i] = gradV_i(i, s, a, c, N, M);
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = y[i] * K_m_squared(i-N, s, c, N);
    }
}

void rhs3(int N, int M, int c[], double y[], double result[]){
    //Central potential and K_m squared
    double *s = y;
    double *a = y + N;

    //MaxSAT constants
    double b = 0.0725;
    double alpha = M/N;
    double a_ = 0.0;
    for (int i = 0; i < M; i++) { a_ += a[i]; }
    a_ = a_/M;
    double constant_ = M_PI_2*b*alpha*a_;
    
    for (int i = 0; i < N; i++)
    {
        result[i] = gradV_i(i, s, a, c, N, M) + constant_*sin(M_PI*s[i]);
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = y[i] * K_m_squared(i-N, s, c, N);
    }
}

void rhs4(int N, int M, int c[], double y[], double result[]){
    //Central potential
    double *s = y;
    double *a = y + N;

    //MaxSAT constants
    double b = 0.0725;
    double alpha = M/N;
    double a_ = 0.0;
    for (int i = 0; i < M; i++) { a_ += a[i]; }
    a_ = a_/M;
    double constant_ = M_PI_2*b*alpha*a_;
    
    for (int i = 0; i < N; i++)
    {
        result[i] = gradV_i(i, s, a, c, N, M) + constant_*sin(M_PI*s[i]); //switch the order of m, i
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = y[i] * K_m(i-N, s, c, N);
    }
}

void rhs5(int N, int M, int c[], double y[], double result[]){
    //Time reversed
    double *s = y;
    double *a = y + N;
    
    for (int i = 0; i < N; i++)
    {
        result[i] = -gradV_i(i, s, a, c, N, M);
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = -y[i] * K_m_squared(i-N, s, c, N);
    }
}

void rhs7(int N, int M, double lambda, int c[], double y[], double result[]){
    //Memory supression with exponential aux variables
    double *s = y;
    double *z = y + N;
    double summ;
    
    for (int i = 0; i < N; i++)     // Updating the soft-spin variables
    {
        summ = 0.0;
        for (int m = 0; m < M; m++)
        {
            summ += 2*c[m*N + i]*k_mi_K_m(m, i, s, c, N)*exp(z[m]);
        }
        result[i] = summ;
    }
    for (int m = N; m < N+M; m++)     // Updating the auxiliary variables
    {
        result[m] = K_m(m-N, s, c, N) - lambda * z[m];
    }
}

void rhs8(int N, int M, double lambda, int c[], double y[], double result[]){
    //Memory supression with regular aux variables
    double *s = y;
    double *a = y + N;
    
    //MaxSAT constants
    double b = 0.0725;
    double alpha = M/N;
    double a_ = 0.0;
    for (int i = 0; i < M; i++) { a_ += a[i]; }
    a_ = a_/M;
    double constant_ = M_PI_2*b*alpha*a_;

    for (int i = 0; i < N; i++)     // Updating the soft-spin variables
    {
        result[i] = gradV_i2(i, s, a, c, N, M) + constant_*sin(M_PI*s[i]);
    }
    for (int i = N; i < N+M; i++)     // Updating the auxiliary variables
    {
        result[i] = y[i] * ( K_m_squared(i-N, s, c, N) - lambda * log( y[i] ) );
    }
}

void rhs9(int N, int M, int c[], double y[], double result[]){
    //Aux variables not updated
    double *s = y;
    double *a = y + N;
    for (int i = 0; i < N; i++)
    {
        result[i] = gradV_i2(i, s, a, c, N, M);
    }
    for (int i = N; i < N+M; i++)
    {
        result[i] = 0;
    }
}

void rhs10(int N, int M, int c[], double y[], double result[]){

    double *s = y;
    double *b = y + N;

    // Pre-calculating k_m values 
    double k[M];
    for (int m = 0; m < M; m++)
    {
        k[m] = K_m(m, s, c, N);
    }

    // Second order memory EoM
    for (int i = 0; i < N; i++)
    {
        result[i] = second_order_potential_prune(i, s, b, c, N, M);
    }
    for (int i = 0; i < M*M; i++)
    { // Updating b_mn variables
        int n = i % M;
        int m = i/M;
        if (n >= m){
            result[N+i] = y[N+i] * k[m] * k[n];
        }
    }
}

void rhs11(int N, int M, int c[], double y[], double result[]){
    // Second order memory
    double *s = y;
    double *b = y + N;
    for (int i = 0; i < N; i++)
    {
        result[i] = second_order_potential_prune(i, s, b, c, N, M);
    }
    for (int i = 0; i < M; i++)
    { // Updating b_mn variables
        for (int j = 0; j < M; j++)
        {
           if (i >= j)
           {
                result[N + flat_idx(i,j, M)] = result[N + flat_idx(i,j, M)] * K_m(i, s, c, N) * K_m(j, s, c, N);
           }
        }
    }
}

void rhs15(int N, int M, int c[], double y[], int nbr_edge_orders, int edge_orders[], int L, int edges[], double result[]){
    double kmi[M * N];
    double k[M];
    double *s = y;
    double *gamma = y + N;
    precompute_kmi(N, M, c, s, kmi);
    precompute_km(N, M, c, s, k);
    //Flat index that runs over all tensor elements
    int overall_index = 0;

    //Loop that runs over all hyper edges
    for (int alpha = 0; alpha < nbr_edge_orders; alpha++){

        // Get the relevatn indices in an array m_1, m_2, ...
        int m_alpha_vec[edge_orders[alpha]];
        // Calculate the product K_m1 * K_m2 * ...
        double productum = 1.0;

        for (int m_alpha_idx = 0; m_alpha_idx < edge_orders[alpha]; m_alpha_idx++){
            m_alpha_vec[m_alpha_idx] = edges[overall_index]; //This is m_alpha
            productum *= k[edges[overall_index]]; //This is k_{m_alpha}
            overall_index++;
        }
        //Update tensor element vector field
        //d/dt Gamma_alpha = Gamma_alpha * K_m1 * K_m2 * ...
        result[N+alpha] = gamma[alpha] * productum;

        //Update the soft-spin vector field with  each tensor element contribution
        for (int i = 0; i < N; i++)
        {   
            double inner_sum = 0;
            for (int m_alpha_idx = 0; m_alpha_idx < edge_orders[alpha]; m_alpha_idx++)
            {   
                int m_alpha = m_alpha_vec[m_alpha_idx];
                if (c[m_alpha * N + i] != 0){
                                //Gamma_{m_1, m_2, ...} * c_{m1,i} * k_{m1,i} * k_m1
                    inner_sum += gamma[alpha] * c[m_alpha * N + i] * kmi[m_alpha*N + i] * k[m_alpha];
                }
            }
            result[i] += inner_sum;
        }
    }
}

void jacobian1(int N, int M, int c[], double y[], double result[]){
    double *s = y;
    double *a = y + N;
    //J11
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double kronecker_ij = 0.0;
            if (i == j) { kronecker_ij = 1.0; }
            double summ = 0;
            for (int m = 0; m < M; m++)
            {
                double prod = 1;
                for (int l = 0; l < N; l++)
                {
                    if (l != i && l != j)
                    {
                        prod *= (1-c[m*N + l]*s[l]);
                    }
                }
                //2*2^(-2k); for SAT3 only!!!
                summ += 0.03125*a[m]*c[m*N + i]*(-c[m*N + j])*(1-c[m*N + j]*s[j])*(1-c[m*N + i]*s[i]) * (1+kronecker_ij) * prod;
            }
            
            result[i*(N+M) + j] = summ;
        }
    }
    //J12
    for (int i = 0; i < N; i++)
    {
        for (int j = N; j < N+M; j++)
        {
            result[i*(N+M) + j] = 2*c[j*N + i] * k_mi_K_m(j, i, s, c, N);
        }
    }
    //J21
    for (int i = N; i < N+M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*(N+M) + j] = a[i]*(-c[i*N + j])*k_mi(i, j, s, c, N);
        }
    }
    //J22
    for (int i = N; i < N+M; i++)
    {
        for (int j = N; j < N+M; j++)
        {
            if (i == j) {
                result[i*(N+M) + j] = K_m(i, s, c, N);
            }
            else {
                result[i*(N+M) + j] = 0.0;
            }
        }
    }
}

void jacobian2(int N, int M, int c[], double y[], double result[]){
    //NOT YET IMPLEMENTED!!!!
    double *s = y;
    double *a = y + N;
    //J11
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            double kronecker_ij = 0.0;
            if (i == j) { kronecker_ij = 1.0; }
            double summ = 0;
            for (int m = 0; m < M; m++)
            {
                double prod = 1;
                for (int l = 0; l < N; l++)
                {
                    if (l != i && l != j)
                    {
                        prod *= (1-c[m*N + l]*s[l]);
                    }
                }
                //2*2^(-2k); for SAT3 only!!!
                summ += 0.03125*a[m]*c[m*N + i]*(-c[m*N + j])*(1-c[m*N + j]*s[j])*(1-c[m*N + i]*s[i]) * (1+kronecker_ij) * prod;
            }
            
            result[i*(N+M) + j] = summ;
        }
    }
    //J12
    for (int i = 0; i < N; i++)
    {
        for (int j = N; j < N+M; j++)
        {
            result[i*(N+M) + j] = 2*c[j*N + i] * k_mi_K_m(j, i, s, c, N);
        }
    }
    //J21
    for (int i = N; i < N+M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            result[i*(N+M) + j] = a[i]*(-c[i*N + j])*k_mi(i, j, s, c, N);
        }
    }
    //J22
    for (int i = N; i < N+M; i++)
    {
        for (int j = N; j < N+M; j++)
        {
            if (i == j) {
                result[i*(N+M) + j] = K_m(i, s, c, N);
            }
            else {
                result[i*(N+M) + j] = 0.0;
            }
        }
    }
}
