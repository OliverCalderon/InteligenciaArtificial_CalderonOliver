package InteligenciaArtificial.Puzzle25;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public final class NodeUtil {

    private NodeUtil() {}

    public static final int SIZE = 5;
    public static final int N = SIZE * SIZE; // 25
    public static final byte BLANK = 0;

    public static final byte[] GOAL = new byte[N];
    private static final int[] GOAL_ROW = new int[N];
    private static final int[] GOAL_COL = new int[N];

    static {
        for (int i = 0; i < N - 1; i++) {
            GOAL[i] = (byte) (i + 1);
        }
        GOAL[N - 1] = 0;

        for (int pos = 0; pos < N; pos++) {
            int tile = GOAL[pos];
            GOAL_ROW[tile] = pos / SIZE;
            GOAL_COL[tile] = pos % SIZE;
        }
    }

    public static boolean isGoal(byte[] tiles) {
        for (int i = 0; i < N; i++) {
            if (tiles[i] != GOAL[i]) return false;
        }
        return true;
    }

    public static int findBlank(byte[] tiles) {
        for (int i = 0; i < tiles.length; i++) {
            if (tiles[i] == BLANK) return i;
        }
        return -1;
    }

    /**
     * Parse a manual state from 25 numbers (0..24) separated by spaces/commas.
     * Example:
     *  1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 0
     */
    public static byte[] parseTiles(String input) {
        String cleaned = input.trim().replace(',', ' ');
        String[] parts = cleaned.split("\\s+");
        if (parts.length != N) {
            throw new IllegalArgumentException("Se esperaban " + N + " numeros, pero llegaron " + parts.length);
        }
        byte[] tiles = new byte[N];
        boolean[] seen = new boolean[N];

        for (int i = 0; i < N; i++) {
            int v;
            try {
                v = Integer.parseInt(parts[i]);
            } catch (NumberFormatException e) {
                throw new IllegalArgumentException("Valor no valido: '" + parts[i] + "'");
            }
            if (v < 0 || v >= N) {
                throw new IllegalArgumentException("Cada numero debe estar entre 0 y 24. Valor: " + v);
            }
            if (seen[v]) {
                throw new IllegalArgumentException("Numero repetido: " + v);
            }
            seen[v] = true;
            tiles[i] = (byte) v;
        }
        return tiles;
    }

    /**
     * For odd grid width (5), solvable iff number of inversions is even.
     */
    public static boolean isSolvable(byte[] tiles) {
        int inv = inversionCount(tiles);
        return (inv % 2) == 0;
    }

    private static int inversionCount(byte[] tiles) {
        int inv = 0;
        for (int i = 0; i < N; i++) {
            int a = tiles[i] & 0xFF;
            if (a == 0) continue;
            for (int j = i + 1; j < N; j++) {
                int b = tiles[j] & 0xFF;
                if (b == 0) continue;
                if (a > b) inv++;
            }
        }
        return inv;
    }

    public static int manhattan(byte[] tiles) {
        int sum = 0;
        for (int pos = 0; pos < N; pos++) {
            int tile = tiles[pos];
            if (tile == 0) continue;
            int r = pos / SIZE;
            int c = pos % SIZE;
            sum += Math.abs(r - GOAL_ROW[tile]) + Math.abs(c - GOAL_COL[tile]);
        }
        return sum;
    }

    //Conflicto lineal = Manhattan + 2*(No. Conflictos)
    public static int linearConflict(byte[] tiles) {
        int man = manhattan(tiles);
        int conflicts = 0;

        // Fila
        int[] goalCols = new int[SIZE];
        for (int r = 0; r < SIZE; r++) {
            int k = 0;
            for (int c = 0; c < SIZE; c++) {
                int tile = tiles[r * SIZE + c];
                if (tile == 0) continue;
                if (GOAL_ROW[tile] == r) {
                    goalCols[k++] = GOAL_COL[tile];
                }
            }
            conflicts += countInversions(goalCols, k);
        }

        // Columna
        int[] goalRows = new int[SIZE];
        for (int c = 0; c < SIZE; c++) {
            int k = 0;
            for (int r = 0; r < SIZE; r++) {
                int tile = tiles[r * SIZE + c] & 0xFF;
                if (tile == 0) continue;
                if (GOAL_COL[tile] == c) {
                    goalRows[k++] = GOAL_ROW[tile];
                }
            }
            conflicts += countInversions(goalRows, k);
        }

        return man + 2 * conflicts;
    }

    private static int countInversions(int[] arr, int len) {
        int inv = 0;
        for (int i = 0; i < len; i++) {
            for (int j = i + 1; j < len; j++) {
                if (arr[i] > arr[j]) inv++;
            }
        }
        return inv;
    }

    public static MovementType opposite(MovementType m) {
        if (m == null) return null;
        switch (m) {
            case UP: return MovementType.DOWN;
            case DOWN: return MovementType.UP;
            case LEFT: return MovementType.RIGHT;
            case RIGHT: return MovementType.LEFT;
            default: return null;
        }
    }

    /**
     * Returns possible moves for a given blank position, optionally avoiding the direct reverse of lastMove.
     */
    public static MovementType[] validMoves(int blankPos, MovementType lastMove) {
        int r = blankPos / SIZE;
        int c = blankPos % SIZE;

        List<MovementType> moves = new ArrayList<>(4);
        if (r > 0) moves.add(MovementType.UP);
        if (r < SIZE - 1) moves.add(MovementType.DOWN);
        if (c > 0) moves.add(MovementType.LEFT);
        if (c < SIZE - 1) moves.add(MovementType.RIGHT);

        MovementType opp = opposite(lastMove);
        if (opp != null) {
            moves.remove(opp);
        }

        return moves.toArray(new MovementType[0]);
    }

    /**
     * Apply a move IN PLACE by swapping blank with an adjacent tile.
     * Returns the new blank position.
     */
    public static int applyMove(byte[] tiles, int blankPos, MovementType move) {
        int swapPos;
        switch (move) {
            case UP: swapPos = blankPos - SIZE; break;
            case DOWN: swapPos = blankPos + SIZE; break;
            case LEFT: swapPos = blankPos - 1; break;
            case RIGHT: swapPos = blankPos + 1; break;
            default: throw new IllegalArgumentException("Movimiento invalido: " + move);
        }
        byte tmp = tiles[blankPos];
        tiles[blankPos] = tiles[swapPos];
        tiles[swapPos] = tmp;
        return swapPos;
    }

    /**
     * Generate a random solvable instance by scrambling the goal state with valid moves.
     * This guarantees solvable states.
     */
    public static byte[] randomSolvable(int scrambleMoves, long seed) {
        Random rnd = new Random(seed);
        byte[] tiles = GOAL.clone();
        int blankPos = N - 1;
        MovementType lastMove = null;

        for (int i = 0; i < scrambleMoves; i++) {
            MovementType[] moves = validMoves(blankPos, lastMove);
            MovementType chosen = moves[rnd.nextInt(moves.length)];
            blankPos = applyMove(tiles, blankPos, chosen);
            lastMove = chosen;
        }
        return tiles;
    }

    public static void printBoard(byte[] tiles) {
        for (int r = 0; r < SIZE; r++) {
            StringBuilder sb = new StringBuilder();
            for (int c = 0; c < SIZE; c++) {
                int v = tiles[r * SIZE + c] & 0xFF;
                if (v == 0) {
                    sb.append("  __");
                } else {
                    sb.append(String.format("%4d", v));
                }
            }
            System.out.println(sb);
        }
    }

    public static String movesToString(List<MovementType> moves) {
        StringBuilder sb = new StringBuilder();
        for (int i = 0; i < moves.size(); i++) {
            sb.append(moves.get(i));
            if (i < moves.size() - 1) sb.append(' ');
        }
        return sb.toString();
    }
}