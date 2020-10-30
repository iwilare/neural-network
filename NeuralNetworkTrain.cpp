#include <vector>
#include <iostream>
#include <random>
#include <fstream>
#include <map>
#include <sstream>
#include <algorithm>
#include <iomanip> // setw
#include <tuple>

#include "NeuralNetwork.cpp"

using namespace std;

enum ValidationType { HOLDOUT, K_FOLD };
struct Configuration {
    random_device::result_type seed;
    ValidationType validationType;
    size_t maxUnluckyEpochs;
    double validationPercentage;
    double validationPercentageDecreaseEpsilon;
    Configuration(random_device::result_type seed=random_device()(),
                  ValidationType validationType=HOLDOUT,
                  size_t maxUnluckyEpochs=1000,
                  double validationPercentage=0.25,
                  double validationPercentageDecreaseEpsilon=0.005)
        : seed(seed),
          validationType(validationType),
          maxUnluckyEpochs(maxUnluckyEpochs),
          validationPercentage(validationPercentage),
          validationPercentageDecreaseEpsilon(validationPercentageDecreaseEpsilon) {}
};

std::istream& operator>>(std::istream& o, Configuration& c) {
    string command;
    while(o >> command)
        if(command == "validationType") {
            string validationType;
            o >> validationType;
            if(validationType == "HOLDOUT")
                c.validationType = HOLDOUT;
            else if(validationType == "K_FOLD")
                c.validationType = K_FOLD;
        } else if(command == "seed") {
            o >> c.seed;
        } else if(command == "maxUnluckyEpochs") {
            o >> c.maxUnluckyEpochs;
        } else if(command == "validationPercentage") {
            o >> c.validationPercentage;
        } else if(command == "validationPercentageDecreaseEpsilon") {
            o >> c.validationPercentageDecreaseEpsilon;
        }
    return o;
}

std::istream& operator>>(std::istream& o, dataset& d) {
    size_t inputs, outputs;
    o >> inputs >> outputs;
    while(!o.eof()) {
        vector<double> input(inputs);
        vector<double> output(outputs);
        for(size_t i = 0; i < inputs;  i++) o >> input[i];
        for(size_t i = 0; i < outputs; i++) o >> output[i];
        d.emplace_back(input, output);
    }
    d.pop_back();
    return o;
}

struct Hyperconfiguration {
    vector<size_t> topology;
    double lam;
    double eta;
    double alpha;
    double randomRange;
    Hyperconfiguration(vector<size_t> topology={},
                       double lam=0.0,
                       double eta=1.0,
                       double alpha=0.0,
                       double randomRange=0.9)
        : topology(topology), lam(lam), eta(eta), alpha(alpha), randomRange(randomRange) {}
};

std::ostream& operator<<(std::ostream& o, Hyperconfiguration const& c) {
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

std::istream& operator>>(std::istream& o, vector<Hyperconfiguration>& hyperparameters) {
    string line;
    for(string line; getline(o, line); ) {
        istringstream s(line);
        string command;
        Hyperconfiguration conf;
        while(s >> command)
            if(command == "lam") {
                s >> conf.lam;
            } else if(command == "eta") {
                s >> conf.eta;
            } else if(command == "alpha") {
                s >> conf.alpha;
            } else if(command == "top") {
                for(size_t top; s >> top; )
                    conf.topology.push_back(top);
            }
        hyperparameters.push_back(conf);
    }
    return o;
}

class Validation {
public:
    static pair<dataset, dataset> split(dataset data, size_t secondSetSize) {
        auto firstSetSize = data.size() - secondSetSize;
        dataset a(data.begin(), data.begin() + firstSetSize);
        dataset b(              data.begin() + firstSetSize, data.end());
        return make_pair(a, b);
    }
    static tuple<Hyperconfiguration, NeuralNetwork, size_t, double, vector<vector<pair<double, double>>>>
            holdout(Configuration const& c, vector<Hyperconfiguration> const& hyperchoices, dataset data) {
        vector<vector<pair<double, double>>> resultHistory;

        shuffle(data.begin(), data.end(), mt19937(c.seed));

        pair<dataset, dataset> dataSplit = split(data, data.size() * c.validationPercentage);
        auto trainingSet   = dataSplit.first;
        auto validationSet = dataSplit.second;

        cerr << "Seed: " << c.seed << ", training size: " << trainingSet.size() << ", validation size: " << validationSet.size() << endl;

        Hyperconfiguration absoluteBestConfiguration;
        double absoluteBestValidationError = INFINITY;
        NeuralNetwork bestNN;
        size_t bestEpoch = 0;
        for(auto h : hyperchoices) {
            size_t unluckyEpochs = 0;
            size_t epoch = 0;
            double bestValidationError = INFINITY, previousValidationError = INFINITY;
            vector<pair<double, double>> validationGraph;
            cerr << "Evaluating " << h << endl;
            auto nn = NeuralNetwork(h.topology, h.randomRange, c.seed);
            do {
                nn.batchLearning(trainingSet, h.eta, h.lam);
                auto trainingError   = nn.computeMeanSquaredError(trainingSet);
                auto validationError = nn.computeMeanSquaredError(validationSet);
                epoch++;

                //if(previousValidationError - validationError > previousValidationError * c.validationPercentageDecreaseEpsilon) {
                if(validationError < previousValidationError) {
                    bestValidationError = validationError;
                    unluckyEpochs = 0;
                } else
                    unluckyEpochs++;
                previousValidationError = validationError;
                validationGraph.emplace_back(trainingError, validationError);

                if(epoch % 1000 == 0)
                    cerr << " T="  << left << setw(std::numeric_limits<double>::digits10) << trainingError
                         << " V="  << left << setw(std::numeric_limits<double>::digits10) << validationError
                         << " E="  << epoch
                         << endl;
            } while(unluckyEpochs < c.maxUnluckyEpochs);
            if(bestValidationError < absoluteBestValidationError) {
                absoluteBestValidationError = bestValidationError;
                absoluteBestConfiguration = h;
                bestNN = nn;
                bestEpoch = epoch;
            }
            resultHistory.push_back(validationGraph);
        }
        return make_tuple(absoluteBestConfiguration, bestNN, bestEpoch, absoluteBestValidationError, resultHistory);
    }
};

class Control {
public:
    static void start(Configuration const& config, vector<Hyperconfiguration> const& hyperparameters, dataset const& data) {

        auto result = Validation::holdout(config, hyperparameters, data);

        cerr << "Absolute best after " << get<2>(result)
             << " epochs, V="          << get<3>(result)
             << ": " << endl
             << get<0>(result) << endl;
        cerr << get<1>(result);

        //size_t i = 0;
        //for(auto const& g : result.second) {
        //    cerr << i << " " << g.size() << endl;
        //    for(auto const& t : g)
        //        cerr << t.first << ", " << t.second << endl;
        //}
        //cerr << "---" << endl;
        //cerr << result.first << endl;
    }
};

int main(int argc, char const *argv[]) {
    if(argc != 2 && argc != 3) {
        cerr << "Usage: " << argv[0] << " hyperParametersFile.hyp [configuration.config] < dataset.dat > model.nn\n";
        return 1;
    } else {
        vector<string> arguments(argv + 1, argv + argc);

        dataset data;
        Configuration config;
        vector<Hyperconfiguration> hyperparameters;

        if(arguments.size() == 2)
        ifstream(arguments[1]) >> config;
        ifstream(arguments[0]) >> hyperparameters;
        cin                    >> data;

        Control::start(config, hyperparameters, data);
    }
}
