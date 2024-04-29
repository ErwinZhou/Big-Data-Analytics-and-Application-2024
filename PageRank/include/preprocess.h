// Data Preprocessing

#ifndef PAGERANK_PREPROCESS_H
#define PAGERANK_PREPROCESS_H

#include <vector>
#include <string>
#include <filesystem>
#include <fstream>
#include <algorithm>
#include <iostream>
#include "defs.h"

// Read the input file and return the number of nodes
int getNumNodes(const std::string &filename);

// Read the input file (1 edge per line: FromNodeID ToNodeID) and return the adjacency matrix
void getAdjacencyMatrix(const std::string &filename, AdjacencyEntry* &adjacencyMatrix, int numNodes);

// Save the adjacency matrix to a file
bool saveAdjacencyMatrix(const std::string &filename, const AdjacencyEntry* adjacencyMatrix, int numNodes);

// Break adjacency matrix into stripes
void breakIntoStripes(const AdjacencyEntry* adjacencyMatrix, int numNodes, int numStripes, AdjacencyEntry** &stripes);

// Break adjacency matrix into stripes and save to files
void breakIntoStripes(const AdjacencyEntry* adjacencyMatrix, int numNodes, int numStripes, const std::string &saveDir = DEFAULT_TMP_DIR, const std::string &prefix = DEFAULT_FILE_NAME);

#endif //PAGERANK_PREPROCESS_H
