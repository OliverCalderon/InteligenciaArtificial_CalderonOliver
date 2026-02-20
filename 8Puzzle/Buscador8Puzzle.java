package puzzle8;

import java.util.HashSet;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

public class Buscador8Puzzle {

    public Estado inicio;

    public Buscador8Puzzle(Estado inicio) {
        this.inicio = inicio;
    }

    public Estado bfs(String objetivo) {
        if (inicio == null) return null;

        objetivo = objetivo.replace('0', ' ');

        HashSet<String> vistos = new HashSet<>();
        Queue<Estado> cola = new LinkedList<>();

        cola.add(inicio);
        vistos.add(inicio.estado);

        while (!cola.isEmpty()) {
            Estado actual = cola.poll();

            if (actual.estado.equals(objetivo)) {
                return actual;
            }

            List<Estado> vecinos = actual.expandir();
            for (Estado sig : vecinos) {
                if (!vistos.contains(sig.estado)) {
                    vistos.add(sig.estado);
                    cola.add(sig);
                }
            }
        }

        return null;
    }
}