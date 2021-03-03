#include <bits/stdc++.h>

#include "NeuralNetwork.cpp"

#define DEFAULT_SEED -1

using namespace std;

enum ValidationType { EXTERNAL_VALIDATION, HOLDOUT, K_FOLD };

struct Configuration {
    random_device::result_type seed;
    ValidationType validationType;
    size_t maxUnluckyEpochs;
    double validationPercentage;
    double validationPercentageDecreaseEpsilon;
    string externalValidationFile;
    bool isClassification;
    Configuration(random_device::result_type seed=DEFAULT_SEED,
                  ValidationType validationType=HOLDOUT,
                  size_t maxUnluckyEpochs=1000,
                  bool isClassification=false,
                  double validationPercentage=0.25,
                  double validationPercentageDecreaseEpsilon=0.005,
                  string externalValidationFile="")
        : seed(seed != DEFAULT_SEED ? seed : random_device()()),
          isClassification(isClassification),
          validationType(validationType),
          maxUnluckyEpochs(maxUnluckyEpochs),
          validationPercentage(validationPercentage),
          validationPercentageDecreaseEpsilon(validationPercentageDecreaseEpsilon),
          externalValidationFile(externalValidationFile) {}
};

std::istream& operator>>(std::istream& o, Configuration& c) {
    string command;
    while(o >> command)
        if(command == "validationType") {
            string validationType;
            o >> validationType;
            if(validationType == "EXTERNAL_VALIDATION")
                c.validationType = EXTERNAL_VALIDATION;
            else if(validationType == "HOLDOUT")
                c.validationType = HOLDOUT;
            else if(validationType == "K_FOLD")
                c.validationType = K_FOLD;
        }
        else if(command == "seed")                                o >> c.seed;
        else if(command == "maxUnluckyEpochs")                    o >> c.maxUnluckyEpochs;
        else if(command == "validationPercentage")                o >> c.validationPercentage;
        else if(command == "isClassification")                    o >> c.isClassification;
        else if(command == "externalValidationFile")              o >> c.externalValidationFile;
        else if(command == "validationPercentageDecreaseEpsilon") o >> c.validationPercentageDecreaseEpsilon;
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
    double eta;
    double lambda;
    double alpha;
    double randomRange;
    random_device::result_type seed;
    Hyperconfiguration(vector<size_t> topology={},
                       double eta=1.0,
                       double lambda=0.0,
                       double alpha=0.0,
                       double randomRange=0.9,
                       random_device::result_type seed=DEFAULT_SEED)
        : topology(topology), lambda(lambda), eta(eta), alpha(alpha), randomRange(randomRange),
          seed(seed != DEFAULT_SEED ? seed : random_device()()) {}
};

std::ostream& operator<<(std::ostream& o, Hyperconfiguration const& c) {
    o << "{";
    o << "eta="         << c.eta         << ", ";
    o << "lambda="      << c.lambda      << ", ";
    o << "alpha="       << c.alpha       << ", ";
    o << "randomRange=" << c.randomRange << ", ";
    o << "top=[ ";
    for(auto& t : c.topology)
        o << t << " ";
    o << "], ";
    o << "seed=" << c.seed;
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
            if(command == "top") {
                for(size_t top; s >> top; )
                    conf.topology.push_back(top);
            }
            else if(command == "eta")         s >> conf.eta;
            else if(command == "lambda")      s >> conf.lambda;
            else if(command == "alpha")       s >> conf.alpha;
            else if(command == "randomRange") s >> conf.randomRange;
            else if(command == "seed")        s >> conf.seed;
        hyperparameters.push_back(conf);
    }
    return o;
}

class Validation {
public:
    struct ValidationStatus {
        Hyperconfiguration configuration;
        size_t epoch                         = 0;
        double trainingError                 = INFINITY;
        double validationError               = INFINITY;
        size_t trainingClassificationError   = (size_t)-1;
        size_t validationClassificationError = (size_t)-1;
    };

    typedef tuple<ValidationStatus, vector<vector<pair<double, double>>>> ValidationResults;

    static ValidationResults externalValidation(Configuration const& c, vector<Hyperconfiguration> const& hyperchoices, dataset training) {
        dataset validation;
        auto file = ifstream(c.externalValidationFile);

        file >> validation;

        return Validation::validation(c, hyperchoices, training, validation);
    }
    static ValidationResults holdout(Configuration const& c, vector<Hyperconfiguration> const& hyperchoices, dataset data) {
        shuffle(data.begin(), data.end(), mt19937(c.seed));

        auto split = [](dataset data, size_t secondSetSize) -> pair<dataset, dataset> {
            auto firstSetSize = data.size() - secondSetSize;
            dataset a(data.begin(), data.begin() + firstSetSize);
            dataset b(              data.begin() + firstSetSize, data.end());
            return make_pair(a, b);
        };

        pair<dataset, dataset> dataSplit = split(data, data.size() * c.validationPercentage);
        auto training   = dataSplit.first;
        auto validation = dataSplit.second;

        return Validation::validation(c, hyperchoices, training, validation);
    }
    static ValidationResults validation(Configuration const& c, vector<Hyperconfiguration> const& hyperchoices, dataset& training, dataset& validation) {
        vector<vector<pair<double, double>>> resultHistory;

        cerr << "Seed: " << c.seed
             << ", training size: "   << training.size()
             << ", validation size: " << validation.size() << endl;

        ValidationStatus absoluteBest;

        for(auto h : hyperchoices) {
            size_t unluckyEpochs = 0;
            size_t epoch         = 0;

            ValidationStatus bestStatus;

            vector<pair<double, double>> validationGraph;

            cerr << "Evaluating " << h << endl;
            auto nn = NeuralNetwork(h.topology, h.randomRange, h.seed != DEFAULT_SEED ? h.seed : c.seed != DEFAULT_SEED ? c.seed : DEFAULT_SEED);

            if(!nn.compatible(training)) {
                cerr << "Topology in hyperconfigurations incompatible with training set, received: ("  << training[0].first.size() << "->" << training[0].second.size() << "), expected (" << nn.inputSize() << "->" << nn.outputSize() << ")." << endl;
                exit(1);
            }
            if(!nn.compatible(validation)) {
                cerr << "Topology in hyperconfigurations incompatible with validation set, received: ("  << validation[0].first.size() << "->" << validation[0].second.size() << "), expected (" << nn.inputSize() << "->" << nn.outputSize() << ")." << endl;
                exit(1);
            }

            do {
                epoch++;
                nn.batchLearning(training, h.eta, h.lambda, h.alpha);

                ValidationStatus status {
                    .configuration                 = h,
                    .epoch                         = epoch,
                    .trainingError                 = nn.computeMeanSquaredError(training),
                    .validationError               = nn.computeMeanSquaredError(validation),
                    .trainingClassificationError   = !c.isClassification ? 0 : nn.computeClassificationError(training),
                    .validationClassificationError = !c.isClassification ? 0 : nn.computeClassificationError(validation)
                };

                if(status.validationError < bestStatus.validationError
                   && abs(status.validationError - bestStatus.validationError)
                    > c.validationPercentageDecreaseEpsilon * status.validationError) {
                    bestStatus = status;
                    unluckyEpochs = 0;
                } else
                    unluckyEpochs++;

                validationGraph.emplace_back(status.trainingError, status.validationError);

            } while(unluckyEpochs < c.maxUnluckyEpochs);

            if(bestStatus.validationError < absoluteBest.validationError)
                absoluteBest = bestStatus;

            resultHistory.push_back(validationGraph);
        }
        return make_tuple(absoluteBest, resultHistory);
    }
};

std::ostream& operator<<(std::ostream& o, Validation::ValidationStatus const& s) {
    o << "{";
    o << "config="  << s.configuration << "," << endl;
    o << "epochs="  << s.epoch << ",";
    o << "T="       << s.trainingError << ",";
    o << "V="       << s.validationError;
    if(s.trainingClassificationError != -1) {
        o << ",";
        o << "Tclass="  << s.trainingClassificationError << ",";
        o << "Vclass="  << s.validationClassificationError;
    }
    o << "}";
    return o;
}

class Control {
public:
    static void start(Configuration const& config, vector<Hyperconfiguration> const& hyperparameters, dataset const& data) {
        tuple<Validation::ValidationStatus, vector<vector<pair<double, double>>>> results;
        switch(config.validationType) {
            case HOLDOUT: {
                results = Validation::holdout(config, hyperparameters, data);
                break;
            }
            case EXTERNAL_VALIDATION: {
                results = Validation::externalValidation(config, hyperparameters, data);
                break;
            }
        }
        cerr << "Absolute best" << endl;
        cerr << get<0>(results);
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
        cin >> data;

        Control::start(config, hyperparameters, data);
    }
}
