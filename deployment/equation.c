/* Auto-generated C residual evaluator with exact gplearn protected operators */
#include <math.h>

static inline double p_add(double a, double b) { return a + b; }
static inline double p_sub(double a, double b) { return a - b; }
static inline double p_mul(double a, double b) { return a * b; }
static inline double p_neg(double a) { return -a; }
static inline double p_div(double a, double b) { return (fabs(b) > 0.001) ? (a / b) : 1.0; }
static inline double p_log(double a) { return (fabs(a) > 0.001) ? log(fabs(a)) : 0.0; }
static inline double p_sqrt(double a) { return sqrt(fabs(a)); }

double evaluate_residual(const double* x) {
    double val = p_add(p_sub(x[1], p_mul(p_div(x[0], x[0]), p_sub(x[2], p_sub(p_sub(x[1], p_mul(p_div(x[0], x[0]), p_add(x[4], x[4]))), p_add(p_add(x[2], x[0]), 0.5620799179779252))))), p_sub(p_add(x[3], p_add(x[3], p_sub(p_sub(x[1], p_div(x[0], x[0])), p_add(x[2], x[0])))), p_sqrt(p_sub(p_log(-0.9825416231640118), p_neg(p_neg(x[3]))))));
    if (val < -30.0) val = -30.0;
    if (val > 30.0) val = 30.0;
    return val;
}

void evaluate_residual_batch(const double* X, double* out, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = evaluate_residual(&X[i * 5]);
    }
}
