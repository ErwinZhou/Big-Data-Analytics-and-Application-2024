// PageRank

#ifndef PAGERANK_PAGERANK_H
#define PAGERANK_PAGERANK_H

#include "defs.h"
#include <iostream>
#include <filesystem>
#include <fstream>

// PageRank Algorithm
void pageRank(
        const AdjacencyEntry* adjacencyMatrix,
        int numNodes,
        double beta,
        double* &pageRankScores,
        double tolerance = DEFAULT_TOLERANCE
        );

bool savePageRankScores(
        const std::string &filename,
        const double* pageRankScores,
        int numNodes
        );

#endif //PAGERANK_PAGERANK_H
