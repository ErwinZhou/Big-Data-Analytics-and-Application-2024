// main

#include <iostream>
#include <string>
#include "defs.h"
#include "preprocess.h"
#include "pagerank.h"

inline void help(const char* program) {
    std::cerr << "Usage: " << program << " [-beta <beta>] [-block <numBlocks>]" << std::endl;
    std::cerr << "Options:" << std::endl;
    std::cerr << "  -beta <beta>        Damping factor (default: 0.85)" << std::endl;
    std::cerr << "  -block <numBlocks>  Number of blocks (default: using Basic Update Algorithm)" << std::endl;
}


int main(int argc, char* argv[]) {
    std::string filename = "Data.txt";

    // Help Info
    std::cout << "----- PageRank Help Info -----" << std::endl;
    help(argv[0]);
    std::cout << "------------------------------" << std::endl;

    // Arguments Parsing
    int numBlocks = 0; // 0: Basic Update Algorithm; positive integer: Block-Stripe Update Algorithm
    double beta = DEFAULT_DAMPING_FACTOR;
    if (argc > 1) {
        if (argc == 3 || argc == 5) {
            for (int i = 1; i < argc; i += 2) {
                if (std::string(argv[i]) == "-beta") {
                    beta = std::stod(argv[i + 1]);
                    if (beta <= 0.0 || beta >= 1.0) {
                        std::cerr << "Error: Damping factor should be in (0, 1)" << std::endl;
                        return 1;
                    }
                } else if (std::string(argv[i]) == "-block") {
                    numBlocks = std::stoi(argv[i + 1]);
                    if (numBlocks <= 0) {
                        std::cerr << "Error: Number of blocks should be a positive integer" << std::endl;
                        return 1;
                    }
                } else {
                    help(argv[0]);
                    return 1;
                }
            }
        } else {
            help(argv[0]);
            return 1;
        }
    }

    if (numBlocks == 0) {
        // Basic Update Algorithm
        std::cout << "Calculate PageRank Using Basic Update Algorithm..." << std::endl;
    } else {
        // Block-Stripe Update Algorithm
        std::cout << "Calculate PageRank Using Block-Stripe Update Algorithm..." << std::endl;
        std::cout << "Number of Blocks: " << numBlocks << std::endl;
    }

    std::cout << "Beta: " << beta << std::endl;

    // Data Preprocessing
    std::cout << "Getting Total Number of Nodes..." << std::endl;
    int numNodes = getNumNodes(filename);
    std::cout << "Number of Nodes: " << numNodes << std::endl;

    std::cout << "Getting Adjacency Matrix...";
    AdjacencyEntry *adjacencyMatrix = new AdjacencyEntry[numNodes + 1];
    getAdjacencyMatrix(filename, adjacencyMatrix, numNodes);
    std::cout << "\tDone" << std::endl;

    std::filesystem::path baseDir = DEFAULT_TMP_DIR;
    std::string savePath;

    std::cout << "Saving Adjacency Matrix..." << std::endl;
    savePath = (baseDir / DEFAULT_FILE_NAME).string();
    saveAdjacencyMatrix(savePath, adjacencyMatrix, numNodes);
    std::cout << "Adjacency Matrix Saved to " << savePath << std::endl;

    // PageRank
    double *pageRankScores = nullptr;

    if (numBlocks == 0) {
        // Basic Update Algorithm
        pageRank(adjacencyMatrix, numNodes, beta, pageRankScores, DEFAULT_TOLERANCE);
    } else {
        // Block-Stripe Update Algorithm
        std::cout << "Break Adjacency Matrix into Stripes...";
        breakIntoStripes(adjacencyMatrix, numNodes, numBlocks, DEFAULT_TMP_DIR, DEFAULT_FILE_NAME);
        std::cout << "\tDone"  << std::endl;

        pageRankByBlockStripe(DEFAULT_TMP_DIR, DEFAULT_FILE_NAME, numNodes, beta, numBlocks, pageRankScores, DEFAULT_TOLERANCE);
    }

    // Save Results
    std::cout << "Saving PageRank Scores..." << std::endl;
    savePath = "PageRankScores.txt";
    savePageRankScores(savePath, pageRankScores, numNodes);
    std::cout << "PageRank Scores Saved to "<< savePath << std::endl;

    // Sort and Save Results
    std::cout << "Sorting and Saving PageRank Scores..." << std::endl;
    savePath = "SortedPageRankScores.txt";
    sortAndSavePageRankScores(savePath, pageRankScores, numNodes);
    std::cout << "Sorted PageRank Scores Saved to " << savePath << std::endl;

    delete[] adjacencyMatrix;
    delete[] pageRankScores;

    return 0;
}
