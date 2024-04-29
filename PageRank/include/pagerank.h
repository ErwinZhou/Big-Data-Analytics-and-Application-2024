// PageRank

#ifndef PAGERANK_PAGERANK_H
#define PAGERANK_PAGERANK_H

#include "defs.h"
#include <iostream>
#include <filesystem>
#include <fstream>
#include <sstream>

// PageRank Using Basic Update Algorithm
void pageRank(
        const AdjacencyEntry* adjacencyMatrix,
        int numNodes,
        double beta,
        double* &pageRankScores,
        double tolerance = DEFAULT_TOLERANCE
        );

// PageRank Using Block-Stripe Update Algorithm
void pageRankByBlockStripe(
        const std::string &dir,
        const std::string &prefix,
        int numNodes,
        double beta,
        int numBlocks,
        double* &pageRankScores,
        double tolerance = DEFAULT_TOLERANCE
        );

// Save PageRank Scores
bool savePageRankScores(
        const std::string &filename,
        const double* pageRankScores,
        int numNodes
);

#endif //PAGERANK_PAGERANK_H
