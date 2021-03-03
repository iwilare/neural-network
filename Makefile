CXX=g++
CXXFLAGS=
all: base
clean:
	rm -rf out/*

base: out/nn-run out/nn-train

debug: CXXFLAGS+=-DDEBUG -g
debug: out/nn-run out/nn-train

release: CXXFLAGS+=-O3
release: out/nn-run out/nn-train

out/nn-run: src/NeuralNetworkRun.cpp src/NeuralNetwork.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
out/nn-train: src/NeuralNetworkTrain.cpp src/NeuralNetwork.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
out/monk-converter: src/MonkConverter.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
debug-run: debug nn-run
	gdb nn-run
debug-train: debug nn-train
	gdb nn-train

convert-monk: out/monk-converter
	./out/monk-converter < monks/monks-1.test  > monks-data/monks-1.test-data
	./out/monk-converter < monks/monks-2.test  > monks-data/monks-2.test-data
	./out/monk-converter < monks/monks-3.test  > monks-data/monks-3.test-data
	./out/monk-converter < monks/monks-1.train > monks-data/monks-1.train-data
	./out/monk-converter < monks/monks-2.train > monks-data/monks-2.train-data
	./out/monk-converter < monks/monks-3.train > monks-data/monks-3.train-data
