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

// Break adjacency matrix into stripes
void breakIntoStripes(const AdjacencyEntry* adjacencyMatrix, int numNodes, int numStripes, AdjacencyEntry** &stripes) {
    if (stripes == nullptr) {
        stripes = new AdjacencyEntry*[numStripes];
    }

    const int stripeSize = (numNodes + numStripes - 1) / numStripes;
    for (int i = 0; i < numStripes; i++) {
        stripes[i] = new AdjacencyEntry[stripeSize + 1];
    }

    for (int i = 1; i <= numNodes; i++) {
        const int outDegree = adjacencyMatrix[i].outDegree;
        int stripeIndex = -1, boundary = 0;
        for (int j = 0; j < outDegree; j++) {
            auto element = adjacencyMatrix[i].outLinks[j];
            while (stripeIndex < (numStripes - 1) && element > boundary) {
                stripeIndex++;
                boundary += stripeSize;
                stripes[stripeIndex][i].outDegree = adjacencyMatrix[i].outDegree;
            }
            stripes[stripeIndex][i].outLinks.push_back(element);
        }
    }
}

// Break adjacency matrix into stripes and save to files
void breakIntoStripes(const AdjacencyEntry* adjacencyMatrix, int numNodes, int numStripes, const std::string &saveDir, const std::string &prefix) {
    std::filesystem::path basePath = saveDir;
    if (!std::filesystem::exists(basePath)) {
        std::filesystem::create_directories(basePath);
    }

    // File Array
    auto *fileArr = new std::ofstream[numStripes];

    const int stripeSize = (numNodes + numStripes - 1) / numStripes;
    for (int i = 0; i < numStripes; i++) {
        std::string filename = prefix + std::to_string(i) + ".txt";
        filename = basePath / filename;
        fileArr[i].open(filename);

        if (!fileArr[i].is_open()) {
            std::cerr << "Error: Unable to open file " << filename << std::endl;
            exit(1);
        }
    }

    for (int i = 1; i <= numNodes; i++) {
        const int outDegree = adjacencyMatrix[i].outDegree;
        int stripeIndex = -1, boundary = 0;
        for (int j = 0; j < outDegree; j++) {
            auto element = adjacencyMatrix[i].outLinks[j];
            while (stripeIndex < (numStripes - 1) && element > boundary) {
                stripeIndex++;
                boundary += stripeSize;
                fileArr[stripeIndex] << i << " " << adjacencyMatrix[i].outDegree;
            }
            fileArr[stripeIndex] << " " << element;
        }

        if (outDegree == 0) { // Dead end
            for (int stripeId = 0; stripeId < numStripes; stripeId++) {
                fileArr[stripeId] << i << " " << 0;
                fileArr[stripeId] << std::endl;
            }
        } else {
            for (int stripeId = 0; stripeId < numStripes; stripeId++) {
                fileArr[stripeId] << std::endl;
            }
        }
    }

    // Close files
    for (int i = 0; i < numStripes; i++) {
        fileArr[i].close();
    }

    delete[] fileArr;
}
