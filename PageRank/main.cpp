// main

#include <iostream>
#include <string>
#include "defs.h"
#include "preprocess.h"
#include "pagerank.h"

int main()
{
    std::string filename = "Data.txt";
    std::cout << "Getting Total Number of Nodes..." << std::endl;
    int numNodes = getNumNodes(filename);
    AdjacencyEntry *adjacencyMatrix = new AdjacencyEntry[numNodes + 1];
    std::cout << "Number of Nodes: " << numNodes << std::endl;

    std::cout << "Getting Adjacency Matrix..." << std::endl;
    getAdjacencyMatrix(filename, adjacencyMatrix, numNodes);

    double *pageRankScores = new double[numNodes + 1];
    std::cout << "Calculating PageRank Scores..." << std::endl;
    pageRank(adjacencyMatrix, numNodes, 0.85, pageRankScores, DEFAULT_TOLERANCE);

    std::cout << "Saving PageRank Scores..." << std::endl;
    savePageRankScores("PageRankScores.txt", pageRankScores, numNodes);

    delete[] adjacencyMatrix;
    delete[] pageRankScores;
    return 0;
}
