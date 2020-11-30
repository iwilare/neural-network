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
    if(k == 2) // One neuron is enough for a bi-state
        cout << (v == 2 ? 1 : -1) << " ";
    else
        for(size_t i = 1; i <= k; i++)
            cout << (i == v ? 1 : -1) << " ";
}

int main(int argc, char **argv) {
    // Print dataset structure on the first line
    cout << 3 + 3 + 1 + 3 + 4 + 1 << " " << 1 << endl;

    int out;
    while(readOutput(&out)) {
        string label;
        oneOfK(3);
        oneOfK(3);
        oneOfK(2); // Just one neuron
        oneOfK(3);
        oneOfK(4);
        oneOfK(2); // Just one neuron
        cin >> label;
        cout << out;
        cout << endl;
    }
}
