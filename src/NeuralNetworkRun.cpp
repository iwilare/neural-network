#include <bits/stdc++.h>

#include "NeuralNetwork.cpp"

using namespace std;

class NeuralNetworkInput {
    istream& input;
    size_t inputs;
    class Iterator : iterator<forward_iterator_tag, vector<double>> {
        istream& input;
        size_t inputs;
        bool end;
    public:
        Iterator(istream& input, size_t inputs, bool end) : input(input), inputs(inputs), end(end) {}
        Iterator const& operator++() { return *this; }
        // Strictly not idempotent
        vector<double> operator*() {
            vector<double> pattern(inputs);
            size_t i;
            for(i = 0; input >> pattern[i] && i < inputs; i++)
                ;
            end = i < inputs;
            return pattern;
        }
        bool operator!=(Iterator const& that) {
            return end != that.end;
        }
    };
public:
    NeuralNetworkInput(istream& input, size_t inputs) : input(input), inputs(inputs) {}
    Iterator begin() { return Iterator(input, inputs, false); }
    Iterator end()   { return Iterator(input, inputs, true);  }
};

int main(int argc, char const *argv[]) {
    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " model.nn < input.in > output.out\n";
        return 1;
    } else {
        vector<string> arguments(argv + 1, argv + argc);

        ifstream modelStream(arguments[0]);

        auto model = NeuralNetwork(modelStream);
        auto input = NeuralNetworkInput(cin, model.inputSize());

        for(auto const& data : input) {
            for(auto const& out : model.feedForward(data))
                cout << out << " ";
            cout << endl;
        }
    }
}
