#include <vector>
#include <iostream>
#include <random>

using namespace std;

typedef vector<pair<vector<double>, vector<double>>> dataset;

double squaredError(vector<double> const& output, vector<double> const& target) {
    double sum = 0;
    // We iterate on target because output might have the bias at the end
    for(size_t i = 0; i < target.size(); i++)
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

    vector<double> deltas;
    vector<double> output;

    void allocateInitialize(size_t inputs, size_t outputs) {
        this->inputs = inputs;
        this->outputs = outputs;

        weights.resize(outputs);
        for(size_t i = 0; i < outputs; i++)
            weights[i].resize(inputs + 1);

        weightsAdj.resize(outputs);
        for(size_t i = 0; i < outputs; i++)
            weightsAdj[i].resize(inputs + 1);

        output.resize(outputs + 1);
        output[outputs] = 1.0;

        deltas.resize(outputs);
    }
public:
    Layer(istream& f) {
        f >> inputs >> outputs;

        allocateInitialize(inputs, outputs);

        for(size_t i = 0; i < outputs; i++)
            for(size_t j = 0; j < inputs + 1; j++)
                f >> weights[i][j];
    }
    Layer(size_t inputs, size_t outputs, double randomRange, random_device::result_type seed) {
        allocateInitialize(inputs, outputs);

        mt19937 mt = mt19937(seed);
        uniform_real_distribution<> dis(-randomRange, +randomRange);
        for(size_t i = 0; i < outputs; i++)
            for(size_t j = 0; j < inputs + 1; j++)
                weights[i][j] = dis(mt);
    }
    vector<double> const& currentOutput() const { return output; }
    size_t inputSize() const { return inputs; }
    size_t outputSize() const { return outputs; }
    vector<vector<double>> const& getWeights() const { return weights; }
    void initializeBatchAdjustement() {
        for(size_t i = 0; i < outputs; i++)
            fill(weightsAdj[i].begin(), weightsAdj[i].end(), 0.0);
    }
    void applyBatchAdjustement(double scaling, double eta) {
        for(size_t i = 0; i < outputs; i++) {
            for(size_t j = 0; j < inputs + 1; j++)
               weights[i][j] += weightsAdj[i][j] * eta / scaling;
        }
    }
    // Requires that input has 1.0 at the end
    vector<double> const& feedForward(vector<double> const& input) {
        for(size_t i = 0; i < outputs; i++) {
            double net = 0.0;
            for(size_t j = 0; j < inputs + 1; j++)
                net += weights[i][j] * input[j];
            output[i] = tanh(net);
        }
        return output;
    }
    // Requires that previousLayerOutputs has 1.0 at the end
    void backpropagate(vector<double> const& previousLayerOutputs, vector<double> const& localError, double lam) {
        for(size_t i = 0; i < outputs; i++) {
            deltas[i] = localError[i] * (1 - output[i] * output[i]);
            for(size_t j = 0; j < inputs + 1; j++)
                weightsAdj[i][j] += -previousLayerOutputs[j] * deltas[i]; // Add normalization here
        }
    }
    void backpropagateOutput(Layer const& previousLayer, vector<double> const& target, double lam) {
        vector<double> localError(outputSize());
        for(size_t i = 0; i < outputs; i++)
            localError[i] = (output[i] - target[i]);
        backpropagate(previousLayer.currentOutput(), localError, lam);
    }
    void backpropagateHidden(Layer const& previousLayer, Layer const& nextLayer, double lam) {
        vector<double> localError(outputSize());
        for(size_t i = 0; i < outputs; i++) {
            localError[i] = 0.0;
            for(size_t k = 0; k < nextLayer.outputs; k++)
                localError[i] += nextLayer.deltas[k] * nextLayer.weights[k][i];
        }
        backpropagate(previousLayer.currentOutput(), localError, lam);
    }
    // Requires that input has 1.0 at the end
    void backpropagateInput(vector<double> const& input, Layer& nextLayer, double lam) {
        vector<double> localError(outputSize());
        for(size_t i = 0; i < outputs; i++) {
            localError[i] = 0.0;
            for(size_t k = 0; k < nextLayer.outputs; k++)
                localError[i] += nextLayer.deltas[k] * nextLayer.weights[k][i];
        }
        backpropagate(input, localError, lam);
    }
};

std::ostream& operator<<(std::ostream& o, Layer const& l) {
    o << l.inputSize() << " " << l.outputSize() << endl;
    for(auto const& r : l.getWeights()) {
        for(auto const& v : r)
            o << v << " ";
        o << endl;
    }
    o << endl;
    return o;
}

class NeuralNetwork {
    vector<Layer> layers;
public:
    NeuralNetwork() {}
    NeuralNetwork(vector<size_t> topology, double randomRange, random_device::result_type seed) {
        layers.reserve(topology.size() - 1);
        for(size_t i = 0; i < topology.size() - 1; i++)
            layers.push_back(Layer(topology[i], topology[i+1], randomRange, seed));
    }
    NeuralNetwork(istream& f) {
        size_t size;
        f >> size;
        layers.reserve(size);
        for(size_t i = 0; i < size; i++)
            layers.push_back(Layer(f));
    }
    vector<Layer> const& getLayers() const { return layers; }
    Layer const& inputLayer() const { return *layers.begin(); }
    Layer const& outputLayer() const { return *layers.rbegin(); }
    vector<double> const& currentOutput() const { return outputLayer().currentOutput(); }
    size_t inputSize() const { return inputLayer().inputSize(); }
    size_t outputSize() const { return outputLayer().outputSize(); }
    vector<double> feedForward(vector<double> const& input) {
        vector<double> inputAndBias(input);
        inputAndBias.push_back(1.0);

        auto out = inputAndBias;
        for(auto& layer : layers)
            out = layer.feedForward(out);
        return out;
    }
    void backpropagate(vector<double> const& input, vector<double> const& target, double lam) {
        vector<double> inputAndBias(input);
        inputAndBias.push_back(1.0);

        auto l = layers.size() - 1;
        layers[l].backpropagateOutput(layers[l-1], target, lam);
        for(size_t l = layers.size() - 2; l > 0; l--)
            layers[l].backpropagateHidden(layers[l-1], layers[l+1], lam);
        layers[0].backpropagateInput(inputAndBias, layers[1], lam);
    }
    void batchLearning(dataset const& data, double eta, double lam) {
        for(size_t l = 0; l < layers.size(); l++)
            layers[l].initializeBatchAdjustement();

        for(auto const& pattern : data) {
            feedForward(pattern.first);
            backpropagate(pattern.first, pattern.second, lam);
        }

        for(size_t l = 0; l < layers.size(); l++)
            layers[l].applyBatchAdjustement(data.size(), eta);
    }
    double computeMeanSquaredError(dataset const& data) {
        double totalError = 0.0;
        for(auto const& pattern : data) {
            feedForward(pattern.first);
            totalError += squaredError(outputLayer().currentOutput(), pattern.second);
        }
        return totalError / data.size();
    }
};

std::ostream& operator<<(std::ostream& o, NeuralNetwork const& nn) {
    o << nn.getLayers().size() << endl;
    for(auto& l : nn.getLayers())
        o << l;
    return o;
}
