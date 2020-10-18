#include <vector>
#include <iostream>
#include <random>
#include <fstream>

#include "NeuralNetwork.cpp"

using namespace std;

class Configuration {
public:
    double lam;
    double eta;
    double randomRange;
    vector<size_t> topology;
    Configuration(double lam=0.0, double eta=1.0, double randomRange=0.9, vector<size_t> topology={})
        : lam(lam), eta(eta), randomRange(randomRange), topology(topology) {}
};

class Control {
public:
    static void start(Configuration& config) {
        auto nn = NeuralNetwork(config.topology, -config.randomRange, config.randomRange);
        auto dataset = DataManager::readBatch(cin, nn.dimensions());

        double totalError;
        do {
            totalError = nn.batchLearning(dataset, config.eta, config.lam);
            cout << "Total dataset error: " << totalError << endl;
        } while(totalError > 0.50);
        nn.serialize(cout);
    }
};

int main(int argc, char const *argv[]) {
    if(argc == 1) {
        cerr << "Usage: " << argv[0] << "inputs hidden1 ... hiddenL outputs < dataset > model\n"
                "           [--eta 0.1 (learning rate)]\n"
                "           [--lam 0.0 (normalization)]\n"
                "           [--rr 0.9 (random weights max range)]\n";
        return 1;
    } else {
        vector<string> arguments(argv + 1, argv + argc);
        Configuration config;
        for(size_t i = 0; i < arguments.size(); i++)
            if(arguments[i] == "--lam")
                config.lam = stod(arguments[++i]);
            else if(arguments[i] == "--eta")
                config.eta = stod(arguments[++i]);
            else if(arguments[i] == "--rr")
                config.randomRange = stod(arguments[++i]);
            else
                config.topology.push_back(stoi(arguments[i]));

        Control::start(config);
    }
}
