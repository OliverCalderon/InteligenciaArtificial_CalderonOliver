package InteligenciaArtificial.Puzzle25;

import java.util.Arrays;

/**
 * Node model kept for compatibility with the original project structure.
 *
 * IDA* does not require keeping all nodes in memory, but we keep this class
 * as a convenient container for a state when needed.
 */
public final class Node {
    private final byte[] tiles;   // length = 25, values 0..24 (0 = blank)
    private final int blankPos;   // index of 0 inside tiles

    public Node(byte[] tiles) {
        this.tiles = tiles;
        this.blankPos = NodeUtil.findBlank(tiles);
    }

    public Node(byte[] tiles, int blankPos) {
        this.tiles = tiles;
        this.blankPos = blankPos;
    }

    public byte[] getTiles() {
        return tiles;
    }

    public int getBlankPos() {
        return blankPos;
    }

    @Override
    public String toString() {
        return Arrays.toString(tiles);
    }
}