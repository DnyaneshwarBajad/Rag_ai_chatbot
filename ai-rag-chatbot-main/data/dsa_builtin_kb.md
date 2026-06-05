# Built-in DSA Knowledge Base (Concise)

This is a compact built-in reference for common DSA interview topics. It is intentionally **not exhaustive** and is used as a fallback when the user has not uploaded enough notes.

## Core interview patterns

- Sliding Window: fixed/variable window, maintain invariant; often O(n).
- Two Pointers: converge from ends, partitioning, de-dup.
- Prefix Sum: subarray sums, range queries; hash map for counts.
- Monotonic Stack: next greater/smaller, histogram, span problems.
- Binary Search on Answer: predicate monotonicity; low/high.
- BFS/DFS: shortest unweighted paths (BFS), components (DFS), topo on DAG.
- Greedy: locally optimal with proof; often sorting + invariant.
- DP: define state, transitions, base cases; optimize with rolling arrays.

## Data Structures (high-level)

### Arrays / Strings
- Definition (Array):
  - An **array** is a contiguous block of memory storing **N elements of the same type**, accessed by **index** (0..N-1).
  - Key properties:
    - **O(1)** random access by index (because address = base + index * size)
    - Fixed size in low-level languages; “dynamic arrays” (e.g., Python list, C++ vector) resize by allocating a bigger array and copying.
  - Common interview uses:
    - prefix/suffix computations, two pointers, sliding window, sorting, frequency counting
  - Pitfalls / edge cases:
    - off-by-one, overflow in index math, empty array, duplicates, negative numbers

- Definition (String):
  - A **string** is typically an array (or dynamic array) of characters with additional operations; many problems treat it like an array with constraints.

- Typical operations: scanning, counting, prefix/suffix, sorting, hashing.
- Edge cases: empty, duplicates, negative values, overflow, unicode.

### Linked List
- Patterns: fast/slow pointers, reverse, merge, cycle detection.
- Pitfalls: null handling, pointer updates order.

### Stack / Queue / Deque
- Stack: expression eval, monotonic stack.
- Queue: BFS.
- Deque: sliding window max/min.

### HashMap / HashSet
- Average O(1) insert/find; collisions worst-case.
- Use for frequency, seen-set, prefix sum index, grouping.

### Heap (Priority Queue)
- Top-k, k-way merge, running median (two heaps).
- Complexity: push/pop O(log n).

### Trie
- Prefix queries, dictionary matching, autocomplete.
- Complexity: O(L) per insert/search.

### Trees / BST
- Traversals: inorder, preorder, postorder, level-order.
- BST: inorder gives sorted order; operations average O(log n), worst O(n).

### Graph
- Representation: adjacency list; edges may be directed/undirected, weighted/unweighted.
- Classic: BFS, DFS, Dijkstra, Union-Find, Topological Sort, MST (Kruskal/Prim).

## Algorithms (quick notes)

### Binary Search
- Requires sorted data or monotonic predicate.
- Complexity: O(log n).

### BFS / DFS
- BFS: shortest path in unweighted graphs.
- DFS: components, cycle detection, topo (with colors/stack).
- Complexity: O(V+E).

### Dijkstra
- Non-negative weights only.
- Complexity: O((V+E) log V) with heap.

### Union Find (DSU)
- Path compression + union by rank.
- Amortized nearly O(1).
- Use for connectivity, Kruskal MST.

### Topological Sort
- Only for DAG.
- Kahn’s algorithm (indegree queue) or DFS stack.
- Complexity: O(V+E).

### Floyd-Warshall
- All-pairs shortest paths; handles negative edges (no negative cycles).
- Complexity: O(V^3).

### Kruskal / Prim
- MST for undirected weighted graphs.
- Kruskal: sort edges + DSU.
- Prim: grow tree with heap.

## Complexity reminders

- Sorting: O(n log n)
- HashMap ops: average O(1)
- Two pointers: often O(n)
- DP: depends on state space; optimize memory when possible

## “Prove it” checklist

- State invariant for two pointers/sliding window.
- Show greedy choice is safe (exchange argument).
- For DP: show optimal substructure + overlapping subproblems.
