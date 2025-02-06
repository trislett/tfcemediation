#include <cstdlib>
#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <cmath>
#include <unordered_set>

using namespace std;

template <class RealType>
void tfce(float H, float E, float minT, float deltaT, 
          const vector< vector<int> > & adjacencyList,
          const RealType * __restrict__ image,
          RealType * __restrict__ enhn) {

    int numberOfVertices = adjacencyList.size();

    vector<int> imageI(numberOfVertices);
    for (int i = 0; i < numberOfVertices; ++i) {
        imageI[i] = i;
    }

    sort(imageI.begin(), imageI.end(),
         [&image](int i, int j) {
             return image[i] > image[j];
         });

    RealType maxT = image[imageI[0]];

    if (deltaT == 0) {
        deltaT = maxT / 100;
    }

    vector<int> parent(numberOfVertices);
    vector<int> rank(numberOfVertices, 0);
    for (int i = 0; i < numberOfVertices; ++i) {
        parent[i] = i;
    }

    auto find = [&](int x) {
        while (parent[x] != x) {
            parent[x] = parent[parent[x]]; // Path compression
            x = parent[x];
        }
        return x;
    };

    auto unite = [&](int x, int y) {
        int xroot = find(x);
        int yroot = find(y);
        if (xroot == yroot) return;

        if (rank[xroot] < rank[yroot]) {
            parent[xroot] = yroot;
        } else if (rank[xroot] > rank[yroot]) {
            parent[yroot] = xroot;
        } else {
            parent[yroot] = xroot;
            rank[xroot]++;
        }
    };

    int j = 0;
    for (float T = maxT; T >= minT; T -= deltaT) {
        while (j < numberOfVertices && image[imageI[j]] > T) {
            int current = imageI[j];
            for (int neighbor : adjacencyList[current]) {
                if (image[neighbor] > T) {
                    unite(current, neighbor);
                }
            }
            ++j;
        }

        float HH = pow(T, H);
        vector<float> tfceIncrement(numberOfVertices, 0.0f);

        for (int i = 0; i < numberOfVertices; ++i) {
            if (image[i] > T) {
                int root = find(i);
                tfceIncrement[root] += 1.0f;
            }
        }

        for (int i = 0; i < numberOfVertices; ++i) {
            if (image[i] > T) {
                int root = find(i);
                enhn[i] += pow(tfceIncrement[root], E) * HH;
            }
        }
    }
}
