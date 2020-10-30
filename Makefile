CXX=g++
CXXFLAGS=
all: release
clean:
	rm nn-run nn-train

debug: CXXFLAGS+=-DDEBUG -g
debug: nn-run nn-train

release: CXXFLAGS+=-O3
release: nn-run nn-train

nn-run: NeuralNetworkRun.cpp NeuralNetwork.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
nn-train: NeuralNetworkTrain.cpp NeuralNetwork.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
debug-run: debug nn-run
	gdb nn-run
debug-train: debug nn-train
	gdb nn-train
