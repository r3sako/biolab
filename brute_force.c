#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

int main(int argc, char *argv[]) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s n target w1 w2 ... wn\n", argv[0]);
        return 1;
    }

    int n = atoi(argv[1]);
    long long target = atoll(argv[2]);
    long long weights[32];
    for (int i = 0; i < n; i++) {
        weights[i] = atoll(argv[3 + i]);
    }

    long long total = 1LL << n;
    int count = 0;
    double first_time = -1.0;

    struct timespec ts_start, ts_now;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    long long current_sum = 0;
    long long prev_gray = 0;

    for (long long i = 1; i < total; i++) {
        long long gray = i ^ (i >> 1);
        long long changed = gray ^ prev_gray;
        int bit = __builtin_ctzll(changed);

        if (gray & changed) {
            current_sum += weights[bit];
        } else {
            current_sum -= weights[bit];
        }

        if (current_sum == target) {
            count++;
            if (first_time < 0.0) {
                clock_gettime(CLOCK_MONOTONIC, &ts_now);
                first_time = (ts_now.tv_sec - ts_start.tv_sec) +
                             (ts_now.tv_nsec - ts_start.tv_nsec) / 1e9;
            }
        }

        prev_gray = gray;
    }

    clock_gettime(CLOCK_MONOTONIC, &ts_now);
    double total_time = (ts_now.tv_sec - ts_start.tv_sec) +
                        (ts_now.tv_nsec - ts_start.tv_nsec) / 1e9;

    if (first_time < 0.0) first_time = total_time;

    printf("%.9f %.9f %d\n", first_time, total_time, count);

    return 0;
}
