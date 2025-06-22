// #include <stdio.h>
// #include <stdlib.h>
// #include <stdint.h>
// #include <string.h>
// #include <sys/stat.h>
// #include <sys/types.h>

// #define SYS_N 3488
// #define BATCH_SIZE 50000
// #define NUM_BATCHES 1

// void process_one_bin(const char *bin_filename, const char *txt_filename) {
//     FILE *fin = fopen(bin_filename, "rb");
//     if (!fin) {
//         perror(bin_filename);
//         return;
//     }

//     unsigned char *buffer = malloc(BATCH_SIZE * SYS_N);
//     if (!buffer) {
//         fprintf(stderr, "malloc failed for %s\n", bin_filename);
//         fclose(fin);
//         return;
//     }

//     size_t expected = BATCH_SIZE * SYS_N;
//     if (fread(buffer, 1, expected, fin) != expected) {
//         fprintf(stderr, "Incomplete read from %s\n", bin_filename);
//         free(buffer);
//         fclose(fin);
//         return;
//     }
//     fclose(fin);

//     FILE *fout = fopen(txt_filename, "w");
//     if (!fout) {
//         perror(txt_filename);
//         free(buffer);
//         return;
//     }

//     for (int ct = 0; ct < BATCH_SIZE; ++ct) {
//         int base = ct * SYS_N;
//         fprintf(fout, "Codeword %d:", ct);
//         int any = 0;
//         for (int j = 0; j < SYS_N; ++j) {
//             if (buffer[base + j]) {
//                 fprintf(fout, " %d", j);
//                 any = 1;
//             }
//         }
//         if (!any) fprintf(fout, " (no errors)");
//         fprintf(fout, "\n");
//     }

//     fclose(fout);
//     free(buffer);
// }

// int main() {
//     // Create result/ directory if it doesn't exist
//     struct stat st = {0};
//     if (stat("result", &st) == -1) {
//         mkdir("result", 0755);
//     }

//     char bin_file[64], txt_file[64];
//     for (int i = 0; i < NUM_BATCHES; ++i) {
//         snprintf(bin_file, sizeof(bin_file), "Output/errorstream%d.bin", i);
//         snprintf(txt_file, sizeof(txt_file), "result/%d.txt", i + 1);
//         printf("📥 %s → %s\n", bin_file, txt_file);
//         process_one_bin(bin_file, txt_file);
//     }

//     printf("✅ All .txt files written inside result/ folder.\n");
//     return 0;
// }


#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#define SYS_N 3488
#define BATCH_SIZE 50000
#define NUM_BATCHES 1

void process_one_bin(const char *bin_filename, const char *csv_filename) {
    FILE *fin = fopen(bin_filename, "rb");
    if (!fin) {
        perror(bin_filename);
        return;
    }

    unsigned char *buffer = malloc(BATCH_SIZE * SYS_N);
    if (!buffer) {
        fprintf(stderr, "malloc failed for %s\n", bin_filename);
        fclose(fin);
        return;
    }

    size_t expected = BATCH_SIZE * SYS_N;
    if (fread(buffer, 1, expected, fin) != expected) {
        fprintf(stderr, "Incomplete read from %s\n", bin_filename);
        free(buffer);
        fclose(fin);
        return;
    }
    fclose(fin);

    FILE *fout = fopen(csv_filename, "w");
    if (!fout) {
        perror(csv_filename);
        free(buffer);
        return;
    }

    for (int ct = 0; ct < BATCH_SIZE; ++ct) {
        int base = ct * SYS_N;
        int first = 1;
        fprintf(fout, "%d", ct);
        for (int j = 0; j < SYS_N; ++j) {
            if (buffer[base + j]) {
                fprintf(fout, "%s%d", first ? "," : ",", j);
                first = 0;
            }
        }
        fprintf(fout, "\n");
    }

    fclose(fout);
    free(buffer);
}

int main() {
    struct stat st = {0};
    if (stat("result", &st) == -1) {
        mkdir("result", 0755);
    }

    char bin_file[64], csv_file[64];
    for (int i = 0; i < NUM_BATCHES; ++i) {
        snprintf(bin_file, sizeof(bin_file), "Output/errorstream%d.bin", i);
        snprintf(csv_file, sizeof(csv_file), "result/%d.csv", i + 1);
        printf("📥 %s → %s\n", bin_file, csv_file);
        process_one_bin(bin_file, csv_file);
    }

    printf("✅ All .csv files written inside result/ folder.\n");
    return 0;
}