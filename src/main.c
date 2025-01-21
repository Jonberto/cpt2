#include <stdio.h>
#include <assert.h>

struct token {
    unsigned offset;
    unsigned length;
};

int main() {
    FILE* pf = fopen("data/enc", "r");
    assert(pf);
    fclose(pf);
    printf("test\n");
    return 0;
}