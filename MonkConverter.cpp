#include <iostream>

using namespace std;

bool readOutput(int *out) {
    size_t c;
    bool ok = (bool)(cin >> c);
    *out = c == 1 ? 1 : -1;
    return ok;
}

void oneOfK(size_t k) {
    size_t v; // [1, k]
    cin >> v;
    for(size_t i = 1; i <= k; i++)
        cout << (i == v ? 1 : 0) << " ";
}

int main(int argc, char **argv) {
    // Print dataset structure on the first line
    cout << 3 + 3 + 2 + 3 + 4 + 2 << " " << 1 << endl;

    int out;
    while(readOutput(&out)) {
        string label;
        oneOfK(3);
        oneOfK(3);
        oneOfK(2);
        oneOfK(3);
        oneOfK(4);
        oneOfK(2);
        cin >> label;
        cout << out;
        cout << endl;
    }
}
