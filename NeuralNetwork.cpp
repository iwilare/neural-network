#include <vector>
#include <iostream>
#include <random>

using namespace std;

typedef vector<pair<vector<double>, vector<double>>> dataset;

double squaredError(vector<double> const& output, vector<double> const& target) {
    double sum = 0;
    for(size_t i = 0; i < output.size(); i++)
        sum += (target[i] - output[i]) * (target[i] - output[i]);
    return sum;
}

class Layer {
    /* Assume inputs as column vectors.
       The number of rows is the number of outputs.
       The number of cols is the number of inputs.
       In this sense, a matrix is a function "vec<len=cols> -> vec<len=rows>"
    */
private:
    size_t inputs, outputs;
    vector<vector<double>> weights;
    vector<vector<double>> weightsAdj;
    vector<double>         bias;
    vector<double>         biasAdj;
    vector<double>         deltas;
    vector<double>         output;

    void allocate(size_t inputs, size_t outputs) {
        this->inputs = inputs;
        this->outputs = outputs;

        weights.resize(outputs);
        for(size_t i = 0; i < outputs; i++)
            weights[i].resize(inputs);
        bias.resize(outputs);

        weightsAdj.resize(outputs);
        for(size_t i = 0; i < outputs; i++)
            weightsAdj[i].resize(inputs);
        biasAdj.resize(outputs);

        output.resize(outputs);
        deltas.resize(outputs);
    }
public:
    Layer(istream& f) {
        f >> inputs >> outputs;

        allocate(inputs, outputs);

        for(size_t i = 0; i < outputs; i++)
            for(size_t j = 0; j < inputs; j++)
                f >> weights[i][j];
        for(size_t i = 0; i < outputs; i++)
            f >> bias[i];
    }
    Layer(size_t inputs, size_t outputs, double minRand, double maxRand) {
        allocate(inputs, outputs);

        random_device rd;
        mt19937 gen(rd());
        uniform_real_distribution<> dis(minRand, maxRand);
        for(size_t i = 0; i < outputs; i++)
            for(size_t j = 0; j < inputs; j++) {
                weights[i][j] = dis(gen);
            }
        for(size_t i = 0; i < outputs; i++)
            bias[i] = dis(gen);
        // Biases not initialized to zero
    }
    vector<double> const& currentOutput() const { return output; }
    size_t inputSize() const { return inputs; }
    size_t outputSize() const { return outputs; }
    void initializeBatchAdjustement() {
        for(size_t i = 0; i < outputs; i++)
            fill(weightsAdj[i].begin(), weightsAdj[i].end(), 0.0);
        fill(biasAdj.begin(), biasAdj.end(), 0.0);
    }
    void applyBatchAdjustement(double scaling, double eta) {
        for(size_t i = 0; i < outputs; i++) {
            for(size_t j = 0; j < inputs; j++)
               weights[i][j] += weightsAdj[i][j] * eta / scaling;
            bias[i] += biasAdj[i] * eta / scaling;
        }
    }
    vector<double> const& feedForward(vector<double> const& input) {
        for(size_t i = 0; i < outputs; i++) {
            auto net = bias[i];
            for(size_t j = 0; j < inputs; j++)
                net += weights[i][j] * input[j];
            output[i] = tanh(net);
        }
        return output;
    }
    void backpropagateOutput(Layer& previousLayer, vector<double> const& target, double lam) {
        for(size_t i = 0; i < outputs; i++) {
            deltas[i] = (output[i] - target[i]) * (1 - output[i] * output[i]);

            for(size_t j = 0; j < inputs; j++)
                weightsAdj[i][j] += -previousLayer.output[j] * deltas[i]; // Add normalization here
            biasAdj[i] += -1 * deltas[i];
        }
    }
    void backpropagate(Layer& previousLayer, Layer& nextLayer, double lam) {
        for(size_t i = 0; i < outputs; i++) {
            auto generalError = 0.0;
            for(size_t k = 0; k < nextLayer.outputs; k++)
                generalError += nextLayer.deltas[k] * nextLayer.weights[k][i];
            deltas[i] = generalError * (1 - output[i] * output[i]);

            for(size_t j = 0; j < inputs; j++)
                weightsAdj[i][j] += -previousLayer.output[j] * deltas[i]; // Add normalization here
            biasAdj[i] += -1 * deltas[i];
        }
    }
    void serialize(ostream& f) {
        f << inputs << " " << outputs << endl;
        for(auto const& r : weights) {
            for(auto const& v : r)
                f << v << " ";
            f << endl;
        }
        for(auto const& b : bias)
            f << b << " ";
        f << endl;
    }
};

class NeuralNetwork {
    vector<Layer> layers;
public:
    NeuralNetwork(vector<size_t> topology, double minRand, double maxRand) {
        layers.reserve(topology.size() - 1);
        for(size_t i = 0; i < topology.size() - 1; i++)
            layers.push_back(Layer(topology[i], topology[i+1], minRand, maxRand));
    }
    NeuralNetwork(istream& f) {
        size_t size;
        f >> size;
        layers.reserve(size);
        for(size_t i = 0; i < size; i++)
            layers.push_back(Layer(f));
    }
    Layer const& inputLayer() const { return *layers.begin(); }
    Layer const& outputLayer() const { return *layers.rbegin(); }
    vector<double> const& currentOutput() const { return outputLayer().currentOutput(); }
    pair<size_t, size_t> dimensions() const {
        return make_pair(inputLayer().inputSize(), outputLayer().outputSize());
    }
    vector<double> feedForward(vector<double> const& input) {
        auto out = input;
        for(auto& layer : layers)
            out = layer.feedForward(out);
        return out;
    }
    void backpropagate(vector<double> const& target, double lam) {
        auto l = layers.size() - 1;
        layers[l].backpropagateOutput(layers[l-1], target, lam);
        for(size_t l = layers.size() - 2; l > 0; l--)
            layers[l].backpropagate(layers[l-1], layers[l+1], lam);
    }
    double batchLearning(dataset data, double eta, double lam) {
        for(size_t l = 0; l < layers.size(); l++)
            layers[l].initializeBatchAdjustement();

        double totalError = 0.0;
        for(auto const& pattern : data) {
            feedForward(pattern.first);
            backpropagate(pattern.second, lam);
            totalError += squaredError(outputLayer().currentOutput(), pattern.second);
        }

        for(size_t l = 0; l < layers.size(); l++)
            layers[l].applyBatchAdjustement(data.size(), eta);

        return totalError;
    }
    void serialize(ostream& f) {
        f << layers.size() << endl;
        for(auto& l : layers)
            l.serialize(f);
    }
};

class DataManager {
public:
    static dataset readBatch(istream& s, pair<size_t, size_t> dimensions) {
        auto inputs = dimensions.first, outputs = dimensions.second;
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
    static vector<vector<double>> streamInput(istream& s, pair<size_t, size_t> dimensions) {
        auto inputs = dimensions.first, outputs = dimensions.second;
        vector<vector<double>> data;
        while(!s.eof()) {
            vector<double> input(inputs);
            for(size_t i = 0; i < inputs; i++) s >> input[i];
            data.emplace_back(input);
        }
        return data;
    }
};
