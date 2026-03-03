package InteligenciaArtificial.Puzzle25;

import java.util.Comparator;

public class NodePriorityComparator implements Comparator<Node> {

    @Override
    public int compare(Node x, Node y) {
        int hx = NodeUtil.manhattan(x.getTiles());
        int hy = NodeUtil.manhattan(y.getTiles());
        return Integer.compare(hx, hy);
    }
}