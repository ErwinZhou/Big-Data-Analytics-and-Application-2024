// Defines the constants and types used in the PageRank algorithm.

#ifndef PAGERANK_DEFS_H
#define PAGERANK_DEFS_H

#include <vector>

#define DEFAULT_TOLERANCE 1e-6  // Default convergence threshold
#define DEFAULT_DAMPING_FACTOR 0.85  // Default damping factor (beta)

// Entry for Sparse Adjacency Matrix
struct AdjacencyEntry {
    int outDegree;
    std::vector<int> outLinks;

    AdjacencyEntry() : outDegree(0) {};
};

#endif //PAGERANK_DEFS_H
