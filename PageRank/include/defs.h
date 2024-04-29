// Defines the constants and types used in the PageRank algorithm.

#ifndef PAGERANK_DEFS_H
#define PAGERANK_DEFS_H

#include <vector>

#define DEFAULT_TOLERANCE 1e-20  // Default convergence threshold
#define DEFAULT_DAMPING_FACTOR 0.85  // Default damping factor (beta)
#define DEFAULT_TMP_DIR "tmp"  // Default temporary directory
#define DEFAULT_FILE_NAME "AdjacencyMatrix" // Default file name

// Entry for Sparse Adjacency Matrix
struct AdjacencyEntry {
    int outDegree;
    std::vector<int> outLinks;

    AdjacencyEntry() : outDegree(0) {};
};

#endif //PAGERANK_DEFS_H
