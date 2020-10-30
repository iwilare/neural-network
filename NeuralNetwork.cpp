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
    Layer(size_t inputs, size_t outputs, double randomRange, random_device::result_type seed) {
        allocate(inputs, outputs);

        mt19937 mt = mt19937(seed);
        uniform_real_distribution<> dis(-randomRange, +randomRange);
        for(size_t i = 0; i < outputs; i++)
            for(size_t j = 0; j < inputs; j++) {
                weights[i][j] = dis(mt);
            }
        for(size_t i = 0; i < outputs; i++)
            bias[i] = dis(mt);
        // Biases not initialized to zero
    }
    vector<double> const& currentOutput() const { return output; }
    size_t inputSize() const { return inputs; }
    size_t outputSize() const { return outputs; }
    vector<double> const& getBias() const { return bias; }
    vector<vector<double>> const& getWeights() const { return weights; }
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
    void backpropagate(vector<double> const& previousLayerOutputs, vector<double> const& nextLayerDeltas, double lam) {
        for(size_t i = 0; i < outputs; i++) {
            deltas[i] = nextLayerDeltas[i] * (1 - output[i] * output[i]);
            for(size_t j = 0; j < inputs; j++)
                weightsAdj[i][j] += -previousLayerOutputs[j] * deltas[i]; // Add normalization here
            biasAdj[i] += -1 * deltas[i];
        }
    }
    void backpropagateOutput(Layer& previousLayer, vector<double> const& target, double lam) {
        vector<double> nextLayerDeltas(outputSize());
        for(size_t i = 0; i < outputs; i++)
            nextLayerDeltas[i] = (output[i] - target[i]);
        backpropagate(previousLayer.currentOutput(), nextLayerDeltas, lam);
    }
    void backpropagateHidden(Layer& previousLayer, Layer& nextLayer, double lam) {
        vector<double> nextLayerDeltas(outputSize());
        for(size_t i = 0; i < outputs; i++) {
            nextLayerDeltas[i] = 0.0;
            for(size_t k = 0; k < nextLayer.outputs; k++)
                nextLayerDeltas[i] += nextLayer.deltas[k] * nextLayer.weights[k][i];
        }
        backpropagate(previousLayer.currentOutput(), nextLayerDeltas, lam);
    }
    void backpropagateInput(vector<double> const& input, Layer& nextLayer, double lam) {
        vector<double> nextLayerDeltas(outputSize());
        for(size_t i = 0; i < outputs; i++) {
            nextLayerDeltas[i] = 0.0;
            for(size_t k = 0; k < nextLayer.outputs; k++)
                nextLayerDeltas[i] += nextLayer.deltas[k] * nextLayer.weights[k][i];
        }
        backpropagate(input, nextLayerDeltas, lam);
    }
};

std::ostream& operator<<(std::ostream& o, Layer const& l) {
    o << l.inputSize() << " " << l.outputSize() << endl;
    for(auto const& r : l.getWeights()) {
        for(auto const& v : r)
            o << v << " ";
        o << endl;
    }
    for(auto const& b : l.getBias())
        o << b << " ";
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
        auto out = input;
        for(auto& layer : layers)
            out = layer.feedForward(out);
        return out;
    }
    void backpropagate(vector<double> const& input, vector<double> const& target, double lam) {
        auto l = layers.size() - 1;
        layers[l].backpropagateOutput(layers[l-1], target, lam);
        for(size_t l = layers.size() - 2; l > 0; l--)
            layers[l].backpropagateHidden(layers[l-1], layers[l+1], lam);
        layers[0].backpropagateInput(input, layers[1], lam);
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
