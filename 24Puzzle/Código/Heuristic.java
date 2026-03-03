package InteligenciaArtificial.Puzzle25;

public enum Heuristic {
    MANHATTAN,
    LINEAR_CONFLICT;

    public int estimate(byte[] tiles) {
        switch (this) {
            case MANHATTAN:
                return NodeUtil.manhattan(tiles);
            case LINEAR_CONFLICT:
                return NodeUtil.linearConflict(tiles);
            default:
                throw new IllegalStateException("Unknown heuristic: " + this);
        }
    }
}