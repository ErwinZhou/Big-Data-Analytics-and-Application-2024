// main

#include <iostream>
#include <string>
#include "defs.h"
#include "preprocess.h"
#include "pagerank.h"


int main() {
    std::string filename = "Data.txt";

    std::cout << "Getting Total Number of Nodes..." << std::endl;
    int numNodes = getNumNodes(filename);
    std::cout << "Number of Nodes: " << numNodes << std::endl;

    std::cout << "Getting Adjacency Matrix...";
    AdjacencyEntry* adjacencyMatrix = new AdjacencyEntry[numNodes + 1];
    getAdjacencyMatrix(filename, adjacencyMatrix, numNodes);
    std::cout << "\tDone" << std::endl;

    std::filesystem::path baseDir = DEFAULT_TMP_DIR;
    std::string savePath;

    std::cout << "Saving Adjacency Matrix..." << std::endl;
    savePath = baseDir / DEFAULT_FILE_NAME;
    saveAdjacencyMatrix(savePath, adjacencyMatrix, numNodes);
    std::cout << "Adjacency Matrix Saved to " << savePath << std::endl;

    double* pageRankScores = nullptr;
//    pageRank(adjacencyMatrix, numNodes, DEFAULT_DAMPING_FACTOR, pageRankScores, DEFAULT_TOLERANCE);

    int numBlocks = 1;

    std::cout << "Break Adjacency Matrix into Stripes...";
    breakIntoStripes(adjacencyMatrix, numNodes, numBlocks, DEFAULT_TMP_DIR, DEFAULT_FILE_NAME);
    std::cout << "\tDone"  << std::endl;

    pageRankByBlockStripe(DEFAULT_TMP_DIR, DEFAULT_FILE_NAME, numNodes, DEFAULT_DAMPING_FACTOR, numBlocks, pageRankScores, DEFAULT_TOLERANCE);

    // Save Results
    std::cout << "Saving PageRank Scores..." << std::endl;
    savePath = "PageRankScores.txt";
    savePageRankScores(savePath, pageRankScores, numNodes);
    std::cout << "PageRank Scores Saved to "<< savePath << std::endl;

    delete[] adjacencyMatrix;
    delete[] pageRankScores;

    return 0;
}
