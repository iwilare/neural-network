#include <bits/stdc++.h>

using namespace std;

typedef vector<pair<vector<double>, vector<double>>> dataset;

double squaredError(vector<double> const& output, vector<double> const& target) {
    double sum = 0;
    // We iterate on target because output might have the bias at the end
    for(size_t i = 0; i < target.size(); i++)
        sum += (target[i] - output[i]) * (target[i] - output[i]);
    return sum;
}

vector<int> discretize(vector<double> const& v) {
    vector<int> discrete(v.size());
    transform(v.begin(), v.end(), discrete.begin(), [](auto x) { return x > 0 ? 1 : -1; });
    return discrete;
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
    vector<vector<double>> oldWeightsAdj;

    vector<double> deltas;
    vector<double> output;

    void allocateInitialize(size_t inputs, size_t outputs) {
        this->inputs = inputs;
        this->outputs = outputs;

        weights       = vector<vector<double>>(outputs, vector<double>(inputs + 1));
        weightsAdj    = vector<vector<double>>(outputs, vector<double>(inputs + 1, 0.0));
        oldWeightsAdj = vector<vector<double>>(outputs, vector<double>(inputs + 1));

        output = vector<double>(outputs + 1);
        output[outputs] = 1.0;

        deltas = vector<double>(outputs);
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
    vector<double>         const& currentOutput() const { return output;  }
    vector<vector<double>> const& getWeights()    const { return weights; }
    size_t inputSize()  const { return inputs;  }
    size_t outputSize() const { return outputs; }
    void removeBias() {
        output.pop_back();
    }
    void initializeBatchAdjustement() {
        oldWeightsAdj = weightsAdj;

        for(auto& weightsAdjRow : weightsAdj)
            fill(weightsAdjRow.begin(), weightsAdjRow.end(), 0.0);
    }
    void applyBatchAdjustement(double scaling, double eta, double lambda, double alpha) {
        for(size_t i = 0; i < outputs; i++) {
            for(size_t j = 0; j < inputs + 1; j++) {
                // Include Nesterov momentum; modifying weightsAdj will also impact the next oldWeightsAdj.
                weightsAdj[i][j] = weightsAdj[i][j] * eta / scaling + alpha * oldWeightsAdj[i][j];
                // Include Tikhonov regularization; all the parameters are independent.
                weights[i][j] = weights[i][j] + weightsAdj[i][j] - lambda * weights[i][j];
            }
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
    void backpropagate(vector<double> const& previousLayerOutputs, vector<double> const& localError) {
        for(size_t i = 0; i < outputs; i++) {
            deltas[i] = localError[i] * (1 - output[i] * output[i]);
            for(size_t j = 0; j < inputs + 1; j++)
                weightsAdj[i][j] += -previousLayerOutputs[j] * deltas[i];
        }
    }
    void backpropagateOutput(Layer const& previousLayer, vector<double> const& target) {
        vector<double> localError(outputSize());
        for(size_t i = 0; i < outputs; i++)
            localError[i] = (output[i] - target[i]);
        backpropagate(previousLayer.currentOutput(), localError);
    }
    void backpropagateHidden(Layer const& previousLayer, Layer const& nextLayer) {
        vector<double> localError(outputSize());
        for(size_t i = 0; i < outputs; i++) {
            localError[i] = 0.0;
            for(size_t k = 0; k < nextLayer.outputs; k++)
                localError[i] += nextLayer.deltas[k] * nextLayer.weights[k][i];
        }
        backpropagate(previousLayer.currentOutput(), localError);
    }
    // Requires that input has 1.0 at the end
    void backpropagateInput(vector<double> const& input, Layer& nextLayer) {
        vector<double> localError(outputSize());
        for(size_t i = 0; i < outputs; i++) {
            localError[i] = 0.0;
            for(size_t k = 0; k < nextLayer.outputs; k++)
                localError[i] += nextLayer.deltas[k] * nextLayer.weights[k][i];
        }
        backpropagate(input, localError);
    }
};

std::ostream& operator<<(std::ostream& o, Layer const& l) {
    o << l.inputSize() << " " << l.outputSize() << endl;
    for(auto const& r : l.getWeights()) {
        for(auto const& v : r)
            o << v << " ";
        o << endl;
    }
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
        layers.rbegin()->removeBias();
    }
    NeuralNetwork(istream& f) {
        size_t size;
        f >> size;
        layers.reserve(size);
        for(size_t i = 0; i < size; i++)
            layers.push_back(Layer(f));
        layers.rbegin()->removeBias();
    }
    vector<Layer> const& getLayers() const { return layers; }
    Layer const& inputLayer()   const { return *layers.begin();  }
    Layer const& outputLayer()  const { return *layers.rbegin(); }
    vector<double>              const& currentOutput() const { return outputLayer().currentOutput(); }
    size_t inputSize()          const { return inputLayer().inputSize();   }
    size_t outputSize()         const { return outputLayer().outputSize(); }
    vector<double> const& feedForward(vector<double> const& input) {
        vector<double> inputAndBias(input);
        inputAndBias.push_back(1.0);
        auto out = inputAndBias;
        for(auto& layer : layers)
            out = layer.feedForward(out);
        return outputLayer().currentOutput();
    }
    void backpropagate(vector<double> const& input, vector<double> const& target) {
        vector<double> inputAndBias(input);
        inputAndBias.push_back(1.0);

        auto l = layers.size() - 1;
        layers[l].backpropagateOutput(layers[l-1], target);
        for(size_t l = layers.size() - 2; l > 0; l--)
            layers[l].backpropagateHidden(layers[l-1], layers[l+1]);
        layers[0].backpropagateInput(inputAndBias, layers[1]);
    }
    void batchLearning(dataset const& data, double eta, double lambda, double alpha) {
        for(size_t l = 0; l < layers.size(); l++)
            layers[l].initializeBatchAdjustement();

        for(auto const& pattern : data) {
            feedForward(pattern.first);
            backpropagate(pattern.first, pattern.second);
        }

        for(size_t l = 0; l < layers.size(); l++)
            layers[l].applyBatchAdjustement(data.size(), eta, lambda, alpha);
    }
    double computeMeanSquaredError(dataset const& data) {
        double totalError = 0.0;
        for(auto const& pattern : data) {
            feedForward(pattern.first);
            totalError += squaredError(outputLayer().currentOutput(), pattern.second);
        }
        return totalError / data.size();
    }
    vector<int> classify(vector<double> const& input) {
        return discretize(feedForward(input));
    }
    // double is so that the compute**Error methods are signature-compatible
    double computeClassificationError(dataset const& data) {
        size_t errors = 0;
        for(auto const& pattern : data) {
            auto output = classify(pattern.first);
            if(output != discretize(pattern.second))
                errors++;
        }
        return errors;
    }
    bool compatible(dataset const& data) {
        return data.size() == 0 || data[0].first.size()  == inputSize()
                                && data[0].second.size() == outputSize();
    }
};

std::ostream& operator<<(std::ostream& o, NeuralNetwork const& nn) {
    o << nn.getLayers().size() << endl;
    for(auto& l : nn.getLayers())
        o << l;
    return o;
}
