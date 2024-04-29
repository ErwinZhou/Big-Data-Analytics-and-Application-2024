// Data Preprocessing
#include "preprocess.h"

// Read the input file and return the number of nodes
int getNumNodes(const std::string &filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }
    int fromNode, toNode, numNodes = 0;
    while (file >> fromNode >> toNode) {
        numNodes = std::max(numNodes, std::max(fromNode, toNode));
    }
    file.close();
    return numNodes;
}

// Read the input file (1 edge per line: FromNodeID ToNodeID) and return the adjacency matrix
void getAdjacencyMatrix(const std::string &filename, AdjacencyEntry* &adjacencyMatrix, int numNodes) {
    // NOTE: adjacencyMatrix should be initialized outside this function using new AdjacencyEntry[numNodes + 1]
    if (adjacencyMatrix == nullptr) {
        std::cerr << "Error: Adjacency Matrix NOT Initialized" << std::endl;
        exit(1);
    }

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        exit(1);
    }

    int fromNode, toNode;
    while (file >> fromNode >> toNode) {
        adjacencyMatrix[fromNode].outDegree++;
        adjacencyMatrix[fromNode].outLinks.push_back(toNode);
    }
    file.close();

    // Sort the outLinks for each node
    for (int i = 1; i <= numNodes; i++) {
        std::sort(adjacencyMatrix[i].outLinks.begin(), adjacencyMatrix[i].outLinks.end());
    }
}

// Save the adjacency matrix to a file
bool saveAdjacencyMatrix(const std::string &filename, const AdjacencyEntry* adjacencyMatrix, int numNodes) {
    std::filesystem::path p = filename;
    if (!p.parent_path().empty() && !std::filesystem::exists(p.parent_path())) {
        // Create the directory(s) if not exist
        std::filesystem::create_directories(p.parent_path());
    }
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return false;
    }
    for (int i = 1; i <= numNodes; i++) {
        file << i << " " << adjacencyMatrix[i].outDegree;
        for (int j = 0; j < adjacencyMatrix[i].outDegree; j++) {
            file << " " << adjacencyMatrix[i].outLinks[j];
        }
        file << std::endl;
    }
    file.close();
    return true;
}
