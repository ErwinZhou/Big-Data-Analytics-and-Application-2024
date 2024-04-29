// PageRank
#include "pagerank.h"

void
pageRank(const AdjacencyEntry *adjacencyMatrix, int numNodes, double beta, double *&pageRankScores, double tolerance) {
    if (pageRankScores == nullptr) {
        std::cerr << "Error: PageRank Scores NOT Initialized" << std::endl;
        exit(1);
    }

    // Initialize PageRank Scores
    for (int i = 1; i <= numNodes; i++) {
        pageRankScores[i] = 1.0 / numNodes;
    }

    // Update PageRank Scores Iteratively
    double *newPageRankScores = new double[numNodes + 1];
    const double randomJump = (1.0 - beta) / numNodes;
    double delta = tolerance;
    int iteration = 0;
    while (delta >= tolerance) {
        delta = 0.0;

        for (int i = 1; i <= numNodes; i++) {
            newPageRankScores[i] = 0.0;
        }

        for (int i = 1; i <= numNodes; i++) {
            const int outDegree = adjacencyMatrix[i].outDegree;
            for (int j = 0; j < outDegree; j++) {
                newPageRankScores[adjacencyMatrix[i].outLinks[j]] += pageRankScores[i] / outDegree;
            }
            if (outDegree == 0) { // Dead end
                for (int j = 1; j <= numNodes; j++) {
                    newPageRankScores[j] += pageRankScores[i] / numNodes;
                }
            }
        }

        for (int i = 1; i <= numNodes; i++) {
            newPageRankScores[i] = beta * newPageRankScores[i] + randomJump;
            delta = std::max(delta, std::abs(newPageRankScores[i] - pageRankScores[i]));
        }

        std::copy(newPageRankScores, newPageRankScores + numNodes + 1, pageRankScores);

        std::cout << "Iteration: " << ++iteration << ", Delta: " << delta << std::endl;
    }
    delete[] newPageRankScores;
}

bool savePageRankScores(const std::string &filename, const double *pageRankScores, int numNodes) {
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
        file << i << " " << pageRankScores[i] << std::endl;
    }

    file.close();
    return true;
}
