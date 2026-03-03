package InteligenciaArtificial.Puzzle25;

import java.util.List;
import java.util.Scanner;

public class App {

    public static void main(String[] args) {
        Scanner sc = new Scanner(System.in);

        System.out.println("=== 24-PUZZLE (5x5) ===");

        // 1) Manual o aleatorio
        byte[] start;
        int mode = askInt(sc,
                "Modo:\n" +
                        "  1) Estado manual\n" +
                        "  2) Estado aleatorio resoluble (scramble desde el objetivo)\n" +
                        "> ",
                1, 2);

        if (mode == 1) {
            start = readManualState(sc);
            if (!NodeUtil.isSolvable(start)) {
                System.out.println("\nEse estado NO es resoluble (paridad incorrecta).\n");
                return;
            }
        } else {
            int scramble = askInt(sc,
                    "Scramble (recomendado: 20-60 para IDA*): > ",
                    1, 500);
            long seed = System.nanoTime();
            start = NodeUtil.randomSolvable(scramble, seed);
            System.out.println("\nSeed usado: " + seed);
        }

        System.out.println("\nEstado inicial:");
        NodeUtil.printBoard(start);

        // ✅ Menú SOLO heurísticas
        int option = askInt(sc,
                "\nElige una opcion:\n" +
                        "  1) IDA* (Manhattan)\n" +
                        "  2) IDA* (Conflicto Lineal)\n" +
                        "  3) Tabla comparativa (ambas)\n" +
                        "> ",
                1, 3);

        switch (option) {
            case 1: { // IDA* Manhattan
                SearchTree.Result r = runIDA(start, Heuristic.MANHATTAN);
                printOneAndMaybeReplay(sc, start, r);
                break;
            }
            case 2: { // IDA* Linear Conflict
                SearchTree.Result r = runIDA(start, Heuristic.LINEAR_CONFLICT);
                printOneAndMaybeReplay(sc, start, r);
                break;
            }
            case 3: { // Tabla comparativa (solo 2)
                runComparisonHeuristics(sc, start);
                break;
            }
            default:
                break;
        }
    }

    // ---------------- Comparativa (opción 3) ----------------
    private static void runComparisonHeuristics(Scanner sc, byte[] start) {
        System.out.println("\n=== TABLA COMPARATIVA (IDA*) ===");

        SearchTree.Result idaM = runIDA(start, Heuristic.MANHATTAN);
        SearchTree.Result idaLC = runIDA(start, Heuristic.LINEAR_CONFLICT);

        System.out.printf("%-10s | %-18s | %-12s | %14s | %10s | %12s\n",
                "Algoritmo", "Heurística", "Estado", "Nodos exp.", "Tiempo", "Movimientos");
        System.out.println("-----------+--------------------+--------------+----------------+------------+-------------");

        printRowEs(idaM);
        printRowEs(idaLC);

        // Mostrar una solución (prioriza Manhattan y luego Conflicto Lineal)
        SearchTree.Result show = firstSolved(idaM, idaLC);
        if (show != null) {
            System.out.println("\n=== Solucion mostrada: " + show.algorithm +
                    (show.heuristic != null ? " (" + show.heuristic + ")" : "") + " ===");
            System.out.println("Movimientos (" + show.solutionLength + "):");
            System.out.println(NodeUtil.movesToString(show.moves));

            int replay = askInt(sc, "\nImprimir tablero paso a paso? (1=Si, 2=No) > ", 1, 2);
            if (replay == 1) replaySolution(start, show.moves);
        } else {
            System.out.println("\nNinguna heurística encontró solución (o se alcanzó algún límite interno).");
        }
    }

    private static SearchTree.Result firstSolved(SearchTree.Result... results) {
        for (SearchTree.Result r : results) {
            if (r != null && r.solved) return r;
        }
        return null;
    }

    private static void printRowEs(SearchTree.Result r) {
        String heur = (r.heuristic == null ? "-" : r.heuristic.toString());
        String moves = r.solved ? String.valueOf(r.solutionLength) : "-";
        String estado = traducirEstado(r.status, r.solved);

        System.out.printf("%-10s | %-18s | %-12s | %14d | %9d ms | %12s\n",
                r.algorithm, heur, estado, r.nodesExpanded, r.timeMillis, moves);
    }

    private static String traducirEstado(String status, boolean solved) {
        if (solved) return "RESUELTO";
        if (status == null) return "NO RESUELTO";

        String s = status.trim().toUpperCase();
        switch (s) {
            case "SOLVED": return "RESUELTO";
            case "NOT_SOLVED": return "NO RESUELTO";
            case "NO_SOLUTION": return "SIN SOLUCIÓN";
            case "CUTOFF": return "LÍMITE";
            case "LIMIT": return "LÍMITE";
            case "TIMEOUT": return "TIEMPO";
            default: return status; // si tu SearchTree usa otros textos
        }
    }

    // ---------------- runner ----------------

    private static SearchTree.Result runIDA(byte[] start, Heuristic h) {
        System.out.println("\n[IDA*] Heurística: " + h);
        SearchTree solver = new SearchTree(h);
        SearchTree.Result r = solver.idaStar(start);

        System.out.println("Estado=" + traducirEstado(r.status, r.solved) +
                " | movs=" + (r.solved ? r.solutionLength : "-") +
                " | nodos=" + r.nodesExpanded +
                " | tiempo=" + r.timeMillis + " ms");
        return r;
    }

    // ---------------- replay + I/O helpers ----------------

    private static void printOneAndMaybeReplay(Scanner sc, byte[] start, SearchTree.Result r) {
        if (r != null && r.solved) {
            System.out.println("\nMovimientos (" + r.solutionLength + "):");
            System.out.println(NodeUtil.movesToString(r.moves));

            int replay = askInt(sc, "\nImprimir tablero paso a paso? (1=Si, 2=No) > ", 1, 2);
            if (replay == 1) replaySolution(start, r.moves);
        } else {
            System.out.println("\nNo se resolvió. Estado=" + (r != null ? traducirEstado(r.status, false) : "null"));
        }
    }

    private static void replaySolution(byte[] start, List<MovementType> moves) {
        byte[] tiles = start.clone();
        int blank = NodeUtil.findBlank(tiles);

        System.out.println("\nPaso 0:");
        NodeUtil.printBoard(tiles);

        for (int i = 0; i < moves.size(); i++) {
            MovementType m = moves.get(i);
            blank = NodeUtil.applyMove(tiles, blank, m);
            System.out.println("\nPaso " + (i + 1) + " (" + m + "):");
            NodeUtil.printBoard(tiles);
        }
    }

    private static byte[] readManualState(Scanner sc) {
        System.out.println("\nIngresa 25 numeros (0..24) separados por espacios.");
        System.out.println("Ejemplo del objetivo:");
        System.out.println("1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 0");

        while (true) {
            System.out.print("> ");
            String line = sc.nextLine();
            try {
                return NodeUtil.parseTiles(line);
            } catch (Exception e) {
                System.out.println("Entrada inválida: " + e.getMessage());
            }
        }
    }

    private static int askInt(Scanner sc, String prompt, int min, int max) {
        while (true) {
            System.out.print(prompt);
            String s = sc.nextLine().trim();
            try {
                int v = Integer.parseInt(s);
                if (v < min || v > max) {
                    System.out.println("Valor fuera de rango (" + min + ".." + max + ").");
                    continue;
                }
                return v;
            } catch (NumberFormatException e) {
                System.out.println("Ingresa un número entero.");
            }
        }
    }
}