#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <iterator>

#include "NeuralNetwork.cpp"

using namespace std;

int main(int argc, char const *argv[]) {
    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " model.nn < input.txt > output.txt\n";
        return 1;
    } else {
        vector<string> arguments(argv + 1, argv + argc);

        ifstream modelStream(arguments[0]);

        auto model = NeuralNetwork(modelStream);
        auto input = DataManager::streamInput(cin, model.dimensions());
        for(auto const& data : input) {
            for(auto const& out : model.feedForward(data))
                cout << out << " ";
            cout << endl;
        }
    }
}
