package puzzle8;

public class Main {
    public static void main(String[] args) {

        Estado inicial = new Estado("1238 4765");
        Buscador8Puzzle buscador = new Buscador8Puzzle(inicial);

        Estado solucion = buscador.bfs("1284376 5");

        System.out.println("Estado Inicial: " + solucion.estado);
        System.out.println("Profundidad: " + solucion.profundidad);
        System.out.println("\n");
        solucion.mostrarRuta();
    }
}