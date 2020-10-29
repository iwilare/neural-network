#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <map>
#include <sstream>
#include <algorithm>
#include <iomanip> // setw

#include "NeuralNetwork.cpp"

using namespace std;

dataset readDataset(istream& s) {
    size_t inputs, outputs;
    s >> inputs >> outputs;
    dataset data;
    while(!s.eof()) {
        vector<double> input(inputs);
        vector<double> output(outputs);
        for(size_t i = 0; i < inputs;  i++) s >> input[i];
        for(size_t i = 0; i < outputs; i++) s >> output[i];
        data.emplace_back(input, output);
    }
    data.pop_back();
    return data;
}

class Hyperconfiguration {
public:
    double lam;
    double eta;
    double alpha;
    double randomRange;
    vector<size_t> topology;
    Hyperconfiguration(vector<size_t> topology={},
                       double lam=0.0,
                       double eta=1.0,
                       double alpha=0.0,
                       double randomRange=0.9)
        : topology(topology), lam(lam), eta(eta), alpha(alpha), randomRange(randomRange) {}
};
std::ostream& operator<< (std::ostream& o, Hyperconfiguration const& c) {
    o << "{";
    o << "lam=" << c.lam << ", ";
    o << "eta=" << c.eta << ", ";
    o << "alpha=" << c.alpha << ", ";
    o << "randomRange=" << c.randomRange << ", ";
    o << "top=[ ";
    for(auto& t : c.topology)
        o << t << " ";
    o << "]";
    o << "}";
    return o;
}

vector<Hyperconfiguration> readHyperparameters(ifstream& hyperFile) {
    vector<Hyperconfiguration> hyperparameters;
    string line;
    for(string line; getline(hyperFile, line); ) {
        istringstream s(line);
        string command;
        Hyperconfiguration conf;
        while(s >> command)
            if(command == "lam") {
                double lam; s >> conf.lam;
            } else if(command == "eta") {
                double eta; s >> conf.eta;
            } else if(command == "alpha") {
                double alpha; s >> conf.alpha;
            } else if(command == "top") {
                for(size_t top; s >> top; )
                    conf.topology.push_back(top);
            }
        hyperparameters.push_back(conf);
    }
    return hyperparameters;
}

class Validation {
public:
    static NeuralNetwork optimize(dataset data, Hyperconfiguration const& c, double threshold) {
        auto nn = NeuralNetwork(c.topology, c.randomRange);

        double totalError;
        do {
            nn.batchLearning(data, c.eta, c.lam);
            totalError = nn.computeTotalError(data);
            cerr << "Total dataset error: " << totalError << endl;
            cout << nn << endl;
        } while(totalError > threshold);

        return nn;
    }
    static pair<dataset, dataset> split(dataset data, size_t secondSetSize) {
        auto firstSetSize = data.size() - secondSetSize;
        dataset a(data.begin(), data.begin() + firstSetSize);
        dataset b(              data.begin() + firstSetSize, data.end());
        return make_pair(a, b);
    }
    static pair<Hyperconfiguration, vector<vector<pair<double, double>>>>
            holdout(dataset data, vector<Hyperconfiguration> hyperchoices, size_t maxUnluckyEpochs, double validationPercentage) {
        vector<vector<pair<double, double>>> resultHistory;

        shuffle(data.begin(), data.end(), mt19937(random_device()()));

        pair<dataset, dataset> dataSplit = split(data, data.size() * validationPercentage);
        auto trainingSet   = dataSplit.first;
        auto validationSet = dataSplit.second;

        Hyperconfiguration absoluteBestConfiguration;
        double absoluteBestValidationError = INFINITY;
        for(auto c : hyperchoices) {
            size_t unluckyEpochs = 0;
            size_t epoch = 0;
            double bestValidationError = INFINITY, previousValidationError = INFINITY;
            vector<pair<double, double>> validationGraph;
            cerr << endl << "Evaluating " << c << endl;
            auto nn = NeuralNetwork(c.topology, c.randomRange);
            do {
                nn.batchLearning(trainingSet, c.eta, c.lam);
                auto trainingError   = nn.computeTotalError(trainingSet);
                auto validationError = nn.computeTotalError(validationSet);
                epoch++;

                if(validationError > previousValidationError) {
                    unluckyEpochs++;
                } else if(validationError < previousValidationError) {
                    bestValidationError = validationError;
                    unluckyEpochs = 0;
                }
                previousValidationError = validationError;
                validationGraph.emplace_back(trainingError, validationError);

                cerr << "\rT=" << left << setw(10) << trainingError << " V=" << left << setw(10) << validationError << " E=" << epoch;
            } while(unluckyEpochs < maxUnluckyEpochs);
            if(bestValidationError < absoluteBestValidationError) {
                absoluteBestValidationError = bestValidationError;
                absoluteBestConfiguration = c;
            }
            resultHistory.push_back(validationGraph);
        }
        cerr << "Absolute best with V=" << absoluteBestValidationError << ": " << absoluteBestConfiguration << endl;
        return make_pair(absoluteBestConfiguration, resultHistory);
    }
};

class Control {
public:
    static void start(dataset const& data, vector<Hyperconfiguration> const& hyperparameters) {
        //cout << Validation::optimize(data, hyperparameters[0], 0.001);
        //return;

        auto result = Validation::holdout(data, hyperparameters, 100, 25.0/100.0);

        size_t i = 0;
        for(auto const& g : result.second) {
            cerr << i << " " << g.size() << endl;
            for(auto const& t : g)
                cerr << t.first << ", " << t.second << endl;
        }
        cerr << "---" << endl;
        cerr << result.first << endl;
    }
};

int main(int argc, char const *argv[]) {
    if(argc != 2) {
        cerr << "Usage: " << argv[0] << " hyperParametersFile.hyp < dataset.dat > model.nn\n";
        return 1;
    } else {
        vector<string> arguments(argv + 1, argv + argc);
        ifstream hyperparametersFile(arguments[0]);
        auto hyperparameters = readHyperparameters(hyperparametersFile);
        auto data            = readDataset(cin);

        Control::start(data, hyperparameters);
    }
}
