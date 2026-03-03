package InteligenciaArtificial.Puzzle25;

import java.util.*;

/**
 * Solvers for 24-Puzzle (5x5):
 * - IDA* (required)
 * - BFS, DFS (depth-limited + maxNodes), UCS
 *
 * BFS/UCS can explode in memory for 5x5, so we support maxNodes cutoffs.
 * DFS can also run "forever" with only depth limit, so we add maxNodes cutoff too.
 */
public class SearchTree {

    public enum Algorithm {
        IDA_STAR,
        BFS,
        DFS,
        UCS
    }

    public static final class Result {
        public final boolean solved;
        public final Algorithm algorithm;
        public final Heuristic heuristic; // only meaningful for IDA*
        public final List<MovementType> moves; // null if not solved
        public final int solutionLength; // -1 if not solved
        public final long nodesExpanded;
        public final long timeMillis;

        /** "SOLVED", "CUTOFF", "FAILED" */
        public final String status;

        Result(boolean solved, Algorithm algorithm, Heuristic heuristic,
            List<MovementType> moves, long nodesExpanded, long timeMillis, String status) {
            this.solved = solved;
            this.algorithm = algorithm;
            this.heuristic = heuristic;
            this.moves = moves;
            this.solutionLength = moves == null ? -1 : moves.size();
            this.nodesExpanded = nodesExpanded;
            this.timeMillis = timeMillis;
            this.status = status;
        }
    }

    // ---------- IDA* internals ----------
    private static final int FOUND = -1;

    private final Heuristic heuristic;

    // Zobrist hashing to detect cycles on current DFS path (IDA* / DFS)
    private final long[][] zobrist = new long[NodeUtil.N][NodeUtil.N]; // [pos][tile]

    private long nodesExpandedIDA;
    private MovementType[] pathMoves;
    private MovementType[] solutionMoves;

    public SearchTree(Heuristic heuristic) {
        this.heuristic = heuristic;
        initZobrist();
    }

    // -------------------- Public API --------------------

    public Result idaStar(byte[] startTiles) {
        if (startTiles == null || startTiles.length != NodeUtil.N) {
            throw new IllegalArgumentException("El estado inicial debe ser byte[25].");
        }
        if (!NodeUtil.isSolvable(startTiles)) {
            return new Result(false, Algorithm.IDA_STAR, heuristic, null, 0, 0, "FAILED");
        }

        byte[] tiles = startTiles.clone();
        int blankPos = NodeUtil.findBlank(tiles);
        long rootHash = hash(tiles);

        this.pathMoves = new MovementType[200_000];
        this.solutionMoves = null;
        this.nodesExpandedIDA = 0;

        LongStack pathHashes = new LongStack(1024);
        pathHashes.push(rootHash);

        long t0 = System.nanoTime();

        int bound = heuristic.estimate(tiles);
        while (true) {
            int next = dfsIDA(tiles, blankPos, 0, bound, null, rootHash, pathHashes);
            if (next == FOUND) {
                long t1 = System.nanoTime();
                List<MovementType> moves = new ArrayList<>();
                for (MovementType m : solutionMoves) moves.add(m);
                return new Result(true, Algorithm.IDA_STAR, heuristic, moves,
                        nodesExpandedIDA, (t1 - t0) / 1_000_000L, "SOLVED");
            }
            if (next == Integer.MAX_VALUE) {
                long t1 = System.nanoTime();
                return new Result(false, Algorithm.IDA_STAR, heuristic, null,
                        nodesExpandedIDA, (t1 - t0) / 1_000_000L, "FAILED");
            }
            bound = next;
        }
    }

    /** BFS with a max expanded nodes cutoff. */
    public Result breadthFirstSearch(byte[] startTiles, long maxNodesExpanded) {
        return bfs(startTiles, maxNodesExpanded);
    }

    /** UCS with a max expanded nodes cutoff. */
    public Result uniformCostSearch(byte[] startTiles, long maxNodesExpanded) {
        return ucs(startTiles, maxNodesExpanded);
    }

    /** DFS depth-limited + max nodes cutoff. */
    public Result depthFirstSearch(byte[] startTiles, int maxDepth, long maxNodesExpanded) {
        return dfsDepthLimited(startTiles, maxDepth, maxNodesExpanded);
    }

    // -------------------- BFS --------------------

    private Result bfs(byte[] startTiles, long maxNodesExpanded) {
        if (startTiles == null || startTiles.length != NodeUtil.N) {
            throw new IllegalArgumentException("El estado inicial debe ser byte[25].");
        }
        if (!NodeUtil.isSolvable(startTiles)) {
            return new Result(false, Algorithm.BFS, null, null, 0, 0, "FAILED");
        }

        long t0 = System.nanoTime();
        long expanded = 0;

        ArrayDeque<NodeRec> q = new ArrayDeque<>();
        HashSet<StateKey> visited = new HashSet<>(100_000);

        byte[] start = startTiles.clone();
        int blank = NodeUtil.findBlank(start);

        NodeRec root = new NodeRec(start, blank, 0, null, null, null);
        q.add(root);
        visited.add(StateKey.of(start));

        while (!q.isEmpty()) {
            if (expanded >= maxNodesExpanded) {
                long t1 = System.nanoTime();
                return new Result(false, Algorithm.BFS, null, null, expanded, (t1 - t0) / 1_000_000L, "CUTOFF");
            }

            NodeRec cur = q.removeFirst();
            expanded++;

            if (NodeUtil.isGoal(cur.tiles)) {
                long t1 = System.nanoTime();
                List<MovementType> path = reconstruct(cur);
                return new Result(true, Algorithm.BFS, null, path, expanded, (t1 - t0) / 1_000_000L, "SOLVED");
            }

            MovementType[] moves = NodeUtil.validMoves(cur.blankPos, cur.lastMove);
            for (MovementType m : moves) {
                byte[] next = cur.tiles.clone();
                int nextBlank = NodeUtil.applyMove(next, cur.blankPos, m);

                StateKey key = StateKey.of(next);
                if (visited.add(key)) {
                    q.addLast(new NodeRec(next, nextBlank, cur.g + 1, cur, m, m));
                }
            }
        }

        long t1 = System.nanoTime();
        return new Result(false, Algorithm.BFS, null, null, expanded, (t1 - t0) / 1_000_000L, "FAILED");
    }

    // -------------------- UCS --------------------

    private Result ucs(byte[] startTiles, long maxNodesExpanded) {
        if (startTiles == null || startTiles.length != NodeUtil.N) {
            throw new IllegalArgumentException("El estado inicial debe ser byte[25].");
        }
        if (!NodeUtil.isSolvable(startTiles)) {
            return new Result(false, Algorithm.UCS, null, null, 0, 0, "FAILED");
        }

        long t0 = System.nanoTime();
        long expanded = 0;

        PriorityQueue<NodeRec> pq = new PriorityQueue<>(Comparator.comparingInt(a -> a.g));
        HashMap<StateKey, Integer> bestCost = new HashMap<>(100_000);

        byte[] start = startTiles.clone();
        int blank = NodeUtil.findBlank(start);

        NodeRec root = new NodeRec(start, blank, 0, null, null, null);
        pq.add(root);
        bestCost.put(StateKey.of(start), 0);

        while (!pq.isEmpty()) {
            if (expanded >= maxNodesExpanded) {
                long t1 = System.nanoTime();
                return new Result(false, Algorithm.UCS, null, null, expanded, (t1 - t0) / 1_000_000L, "CUTOFF");
            }

            NodeRec cur = pq.poll();
            expanded++;

            if (NodeUtil.isGoal(cur.tiles)) {
                long t1 = System.nanoTime();
                List<MovementType> path = reconstruct(cur);
                return new Result(true, Algorithm.UCS, null, path, expanded, (t1 - t0) / 1_000_000L, "SOLVED");
            }

            MovementType[] moves = NodeUtil.validMoves(cur.blankPos, cur.lastMove);
            for (MovementType m : moves) {
                byte[] next = cur.tiles.clone();
                int nextBlank = NodeUtil.applyMove(next, cur.blankPos, m);
                int newG = cur.g + 1;

                StateKey key = StateKey.of(next);
                Integer old = bestCost.get(key);
                if (old == null || newG < old) {
                    bestCost.put(key, newG);
                    pq.add(new NodeRec(next, nextBlank, newG, cur, m, m));
                }
            }
        }

        long t1 = System.nanoTime();
        return new Result(false, Algorithm.UCS, null, null, expanded, (t1 - t0) / 1_000_000L, "FAILED");
    }

    // -------------------- DFS depth-limited (+ maxNodes) --------------------

    private Result dfsDepthLimited(byte[] startTiles, int maxDepth, long maxNodesExpanded) {
        if (startTiles == null || startTiles.length != NodeUtil.N) {
            throw new IllegalArgumentException("El estado inicial debe ser byte[25].");
        }
        if (!NodeUtil.isSolvable(startTiles)) {
            return new Result(false, Algorithm.DFS, null, null, 0, 0, "FAILED");
        }

        byte[] tiles = startTiles.clone();
        int blankPos = NodeUtil.findBlank(tiles);
        long rootHash = hash(tiles);

        this.pathMoves = new MovementType[Math.max(16, maxDepth + 5)];
        this.solutionMoves = null;

        LongStack pathHashes = new LongStack(256);
        pathHashes.push(rootHash);

        long t0 = System.nanoTime();

        DfsState st = new DfsState(maxNodesExpanded);
        boolean ok = dfsRec(tiles, blankPos, 0, maxDepth, null, rootHash, pathHashes, st);

        long t1 = System.nanoTime();

        if (!ok) {
            String status = (st.cutoffNodes || st.hitDepthLimit) ? "CUTOFF" : "FAILED";
            return new Result(false, Algorithm.DFS, null, null, st.nodesExpanded, (t1 - t0) / 1_000_000L, status);
        }

        List<MovementType> moves = new ArrayList<>();
        for (MovementType m : solutionMoves) moves.add(m);
        return new Result(true, Algorithm.DFS, null, moves, st.nodesExpanded, (t1 - t0) / 1_000_000L, "SOLVED");
    }

    private static final class DfsState {
        long nodesExpanded = 0;
        boolean hitDepthLimit = false;
        boolean cutoffNodes = false;
        final long maxNodes;

        DfsState(long maxNodes) {
            this.maxNodes = maxNodes;
        }
    }

    private boolean dfsRec(byte[] tiles,
                           int blankPos,
                           int depth,
                           int maxDepth,
                           MovementType lastMove,
                           long hash,
                           LongStack pathHashes,
                           DfsState st) {

        // ✅ corte por nodos
        if (st.nodesExpanded >= st.maxNodes) {
            st.cutoffNodes = true;
            return false;
        }
        st.nodesExpanded++;

        if (NodeUtil.isGoal(tiles)) {
            solutionMoves = Arrays.copyOf(pathMoves, depth);
            return true;
        }
        if (depth >= maxDepth) {
            st.hitDepthLimit = true;
            return false;
        }

        MovementType[] moves = NodeUtil.validMoves(blankPos, lastMove);
        for (MovementType move : moves) {
            int swapPos = neighborIndex(blankPos, move);
            byte movedTile = tiles[swapPos];

            long newHash = updateHash(hash, blankPos, swapPos, movedTile);
            if (pathHashes.contains(newHash)) continue;

            swap(tiles, blankPos, swapPos);
            pathMoves[depth] = move;
            pathHashes.push(newHash);

            if (dfsRec(tiles, swapPos, depth + 1, maxDepth, move, newHash, pathHashes, st)) {
                return true;
            }

            pathHashes.pop();
            swap(tiles, blankPos, swapPos);
        }

        return false;
    }

    // -------------------- IDA* recursion --------------------

    private int dfsIDA(byte[] tiles,
                       int blankPos,
                       int g,
                       int bound,
                       MovementType lastMove,
                       long hash,
                       LongStack pathHashes) {

        int h = heuristic.estimate(tiles);
        int f = g + h;

        if (f > bound) return f;
        if (NodeUtil.isGoal(tiles)) {
            solutionMoves = Arrays.copyOf(pathMoves, g);
            return FOUND;
        }

        int min = Integer.MAX_VALUE;

        MovementType[] moves = NodeUtil.validMoves(blankPos, lastMove);
        for (MovementType move : moves) {
            int swapPos = neighborIndex(blankPos, move);
            byte movedTile = tiles[swapPos];

            long newHash = updateHash(hash, blankPos, swapPos, movedTile);
            if (pathHashes.contains(newHash)) continue;

            swap(tiles, blankPos, swapPos);
            pathMoves[g] = move;
            pathHashes.push(newHash);
            nodesExpandedIDA++;

            int t = dfsIDA(tiles, swapPos, g + 1, bound, move, newHash, pathHashes);
            if (t == FOUND) return FOUND;
            if (t < min) min = t;

            pathHashes.pop();
            swap(tiles, blankPos, swapPos);
        }

        return min;
    }

    // -------------------- Helpers --------------------

    private static List<MovementType> reconstruct(NodeRec goal) {
        ArrayDeque<MovementType> stack = new ArrayDeque<>();
        NodeRec cur = goal;
        while (cur.parent != null) {
            stack.push(cur.moveFromParent);
            cur = cur.parent;
        }
        return new ArrayList<>(stack);
    }

    private static int neighborIndex(int blankPos, MovementType move) {
        switch (move) {
            case UP: return blankPos - NodeUtil.SIZE;
            case DOWN: return blankPos + NodeUtil.SIZE;
            case LEFT: return blankPos - 1;
            case RIGHT: return blankPos + 1;
            default: throw new IllegalArgumentException("Movimiento invalido: " + move);
        }
    }

    private static void swap(byte[] tiles, int i, int j) {
        byte tmp = tiles[i];
        tiles[i] = tiles[j];
        tiles[j] = tmp;
    }

    // ---------- Zobrist hashing ----------
    private void initZobrist() {
        long seed = 0x9E3779B97F4A7C15L;
        for (int pos = 0; pos < NodeUtil.N; pos++) {
            for (int tile = 0; tile < NodeUtil.N; tile++) {
                seed = splitMix64(seed);
                zobrist[pos][tile] = seed;
            }
        }
    }

    private static long splitMix64(long x) {
        long z = (x + 0x9E3779B97F4A7C15L);
        z = (z ^ (z >>> 30)) * 0xBF58476D1CE4E5B9L;
        z = (z ^ (z >>> 27)) * 0x94D049BB133111EBL;
        return z ^ (z >>> 31);
    }

    private long hash(byte[] tiles) {
        long h = 0;
        for (int pos = 0; pos < NodeUtil.N; pos++) {
            int tile = tiles[pos] & 0xFF;
            h ^= zobrist[pos][tile];
        }
        return h;
    }

    private long updateHash(long hash, int blankPos, int swapPos, byte movedTile) {
        int tile = movedTile & 0xFF;
        hash ^= zobrist[blankPos][0];
        hash ^= zobrist[swapPos][tile];
        hash ^= zobrist[blankPos][tile];
        hash ^= zobrist[swapPos][0];
        return hash;
    }

    // ---------- Small internal structures ----------
    private static final class NodeRec {
        final byte[] tiles;
        final int blankPos;
        final int g;

        final NodeRec parent;
        final MovementType moveFromParent;
        final MovementType lastMove;

        NodeRec(byte[] tiles, int blankPos, int g,
                NodeRec parent, MovementType moveFromParent, MovementType lastMove) {
            this.tiles = tiles;
            this.blankPos = blankPos;
            this.g = g;
            this.parent = parent;
            this.moveFromParent = moveFromParent;
            this.lastMove = lastMove;
        }
    }

    private static final class StateKey {
        final byte[] tiles;
        final int hash;

        private StateKey(byte[] tiles) {
            this.tiles = tiles;
            this.hash = Arrays.hashCode(tiles);
        }

        static StateKey of(byte[] tiles) {
            return new StateKey(tiles.clone()); // avoid mutation bugs
        }

        @Override
        public int hashCode() { return hash; }

        @Override
        public boolean equals(Object o) {
            if (this == o) return true;
            if (!(o instanceof StateKey)) return false;
            StateKey other = (StateKey) o;
            return Arrays.equals(this.tiles, other.tiles);
        }
    }

    private static final class LongStack {
        private long[] data;
        private int size;

        LongStack(int initialCap) {
            data = new long[Math.max(16, initialCap)];
            size = 0;
        }

        void push(long v) {
            if (size >= data.length) data = Arrays.copyOf(data, data.length * 2);
            data[size++] = v;
        }

        long pop() {
            return data[--size];
        }

        boolean contains(long v) {
            for (int i = 0; i < size; i++) {
                if (data[i] == v) return true;
            }
            return false;
        }
    }
}