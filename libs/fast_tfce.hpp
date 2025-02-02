#include <cstdlib>
#include <iostream>
#include <list>
#include <set>
#include <string>
#include <vector>
#include <algorithm>

//    Fast TFCE algorithm
//    Copyright (C) 2025  Tristram Lett

//#    This program is free software: you can redistribute it and/or modify
//#    it under the terms of the GNU General Public License as published by
//#    the Free Software Foundation, either version 3 of the License, or
//#    (at your option) any later version.

//#    This program is distributed in the hope that it will be useful,
//#    but WITHOUT ANY WARRANTY; without even the implied warranty of
//#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//#    GNU General Public License for more details.

//#    You should have received a copy of the GNU General Public License
//#    along with this program.  If not, see <http:www.gnu.org/licenses/>.

using namespace std;

template <typename RealType>
void tfce(float H, float E, float minT, float deltaT, 
          const std::vector<std::vector<int>> &adjacencyList,
          const RealType * __restrict__ image,
          RealType * __restrict__ enhn) {

    int numberOfVertices = adjacencyList.size();

    // Use unique_ptr to automatically handle memory cleanup
    std::vector<std::unique_ptr<std::list<int>>> disjointFind(numberOfVertices);
    std::unordered_set<std::list<int>*> disjointSets;

    std::vector<int> imageI(numberOfVertices);
    for (int i = 0; i < numberOfVertices; ++i) {
        imageI[i] = i;
    }

    // Sort indices by image intensity in descending order
    std::sort(imageI.begin(), imageI.end(), 
              [&image](int i, int j) { return image[i] > image[j]; });

    RealType maxT = image[imageI[0]];
    if (deltaT == 0) {
        deltaT = maxT / 100;
    }

    int j = 0;
    for (float T = maxT; T >= minT; T -= deltaT) {
        // Incremental connectivity using a disjoint-set approach
        while (j < numberOfVertices && image[imageI[j]] > T) {
            int v = imageI[j];
            disjointFind[v] = std::make_unique<std::list<int>>();
            disjointFind[v]->push_front(v);

            disjointSets.insert(disjointFind[v].get());

            for (int neighbor : adjacencyList[v]) {
                if (disjointFind[neighbor] && disjointFind[neighbor] != disjointFind[v]) {
                    std::list<int>* neighborSet = disjointFind[neighbor].get();
                    std::list<int>* currentSet = disjointFind[v].get();

                    // Move elements from current set to neighbor set
                    neighborSet->splice(neighborSet->begin(), *currentSet);

                    // Update disjointFind to reflect new ownership
                    for (int element : *neighborSet) {
                        disjointFind[element] = std::move(disjointFind[v]);
                    }

                    disjointSets.erase(currentSet);
                }
            }
            ++j;
        }

        float HH = std::pow(T, H);

        // Apply TFCE increment
        for (auto* component : disjointSets) {
            float tfceIncrement = std::pow(component->size(), E) * HH;
            for (int idx : *component) {
                enhn[idx] += tfceIncrement;
            }
        }
    }

    // Unique pointers automatically free memory, no manual delete needed
}

