// PageRank
#include "pagerank.h"

void pageRank(const AdjacencyEntry *adjacencyMatrix, int numNodes, double beta, double *&pageRankScores, double tolerance) {
    // Initialize PageRank Scores
    if (pageRankScores == nullptr) {
        pageRankScores = new double[numNodes + 1];
    }
    for (int i = 1; i <= numNodes; i++) {
        pageRankScores[i] = 1.0 / numNodes;
    }

    // Update PageRank Scores Iteratively
    double *newPageRankScores = new double[numNodes + 1];
    const double randomJump = (1.0 - beta) / numNodes;
    double delta = tolerance;
    int iteration = 0;
    std::cout << "Calculating PageRank Scores..." << std::endl;
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

        std::cout << "Iteration: " << ++iteration << ", Max Delta: " << delta << std::endl;
    }
    std::cout << "PageRank Done" << std::endl;

    delete[] newPageRankScores;
}

void pageRankByBlockStripe(const std::string &dir, const std::string &prefix, int numNodes, double beta, int numBlocks, double* &pageRankScores, double tolerance) {
    /*
     * NOTE: Here we directly load r_new and r_old to memory for simplicity
     *       while we still load matrix stripe from disk one row at a time due to limited memory.
     */
    // Initialize PageRank Scores
    if (pageRankScores == nullptr) {
        pageRankScores = new double[numNodes + 1];
    }
    for (int i = 1; i <= numNodes; i++) {
        pageRankScores[i] = 1.0 / numNodes;
    }

    // Update PageRank Scores Iteratively Using Block-Stripe Update Algorithm
    double *newPageRankScores = new double[numNodes + 1];
    const double randomJump = (1.0 - beta) / numNodes;
    const int blockSize = (numNodes + numBlocks - 1) / numBlocks;
    double delta = tolerance;
    int iteration = 0;
    std::filesystem::path basePath = dir;

    // Info
    std::cout << "Calculating PageRank Scores..." << std::endl;
    std::cout << "Using Block-Stripe Update Algorithm" << std::endl;
    std::cout << "Number of Blocks: " << numBlocks << std::endl;
    std::cout << "Number of Nodes Per Block: " << blockSize << std::endl;

    while (delta >= tolerance) {
        delta = 0.0;

        for (int blockID = 0; blockID < numBlocks; blockID++) {
            const int boundary = std::min((blockID + 1) * blockSize, numNodes);
            for (int i = blockID * blockSize + 1; i <= boundary; i++) {
                newPageRankScores[i] = 0.0;
            }

            std::string filename = prefix + std::to_string(blockID) + ".txt";
            filename = (basePath / filename).string();
            std::ifstream file(filename);
            if (!file.is_open()) {
                std::cerr << "Error: Unable to open file " << filename << std::endl;
                exit(1);
            }

            std::string line;
            while (std::getline(file, line)) {
                std::istringstream iss(line);
                int sourceID, outDegree;
                iss >> sourceID >> outDegree;
                int destID;
                while (iss >> destID) {
                    newPageRankScores[destID] += pageRankScores[sourceID] / outDegree;
                }
                if (outDegree == 0) { // Dead end
                    for (int i = blockID * blockSize + 1; i <= boundary; i++) {
                        newPageRankScores[i] += pageRankScores[sourceID] / numNodes;
                    }
                }
            }

            file.close();

            for (int i = blockID * blockSize + 1; i <= boundary; i++) {
                newPageRankScores[i] = beta * newPageRankScores[i] + randomJump;
                delta = std::max(delta, std::abs(newPageRankScores[i] - pageRankScores[i]));
            }
        }

        std::copy(newPageRankScores, newPageRankScores + numNodes + 1, pageRankScores);

        std::cout << "Iteration: " << ++iteration << ", Max Delta: " << delta << std::endl;
    }
    std::cout << "PageRank Done" << std::endl;

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
