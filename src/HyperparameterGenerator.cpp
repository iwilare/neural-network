#include <bits/stdc++.h>

#define DEFAULT_SEED -1

using namespace std;

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
    o << "eta "         << c.eta         << " ";
    o << "lambda "      << c.lambda      << " ";
    o << "alpha "       << c.alpha       << " ";
    o << "randomRange " << c.randomRange << " ";
    o << "top ";
    for(auto& t : c.topology)
        o << t << " ";
    return o;
}

int main() {
    for(double eta=0.1;        eta   < 1.0;     eta   += 0.1)
    for(double alpha=0.1;      alpha < 1.0;     alpha += 0.1)
    cout << Hyperconfiguration({17, 2, 1}, eta, 0.0, alpha, 0.4) << endl;
}
