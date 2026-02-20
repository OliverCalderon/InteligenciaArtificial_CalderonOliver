package puzzle8;

import java.util.LinkedList;

public class Estado {

    public String estado;
    public Estado anterior;
    public int profundidad;
    public int costo;

    public Estado(String estado) {
        this(estado, 0, null);
    }

    public Estado(String estado, int profundidad, Estado anterior) {
        this.estado = limpiar(estado);
        this.profundidad = profundidad;
        this.anterior = anterior;
        this.costo = profundidad;
    }

    public LinkedList<Estado> expandir() {
        LinkedList<Estado> lista = new LinkedList<>();

        int posHueco = this.estado.indexOf(' ');
        int nuevoNivel = this.profundidad + 1;

        switch (posHueco) {
            case 0:
                lista.add(new Estado(swaps(this.estado, 0, 1), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 0, 3), nuevoNivel, this));
                break;

            case 1:
                lista.add(new Estado(swaps(this.estado, 1, 0), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 1, 2), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 1, 4), nuevoNivel, this));
                break;

            case 2:
                lista.add(new Estado(swaps(this.estado, 2, 1), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 2, 5), nuevoNivel, this));
                break;

            case 3:
                lista.add(new Estado(swaps(this.estado, 3, 0), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 3, 4), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 3, 6), nuevoNivel, this));
                break;

            case 4:
                lista.add(new Estado(swaps(this.estado, 4, 1), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 4, 3), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 4, 5), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 4, 7), nuevoNivel, this));
                break;

            case 5:
                lista.add(new Estado(swaps(this.estado, 5, 2), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 5, 4), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 5, 8), nuevoNivel, this));
                break;

            case 6:
                lista.add(new Estado(swaps(this.estado, 6, 3), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 6, 7), nuevoNivel, this));
                break;

            case 7:
                lista.add(new Estado(swaps(this.estado, 7, 4), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 7, 6), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 7, 8), nuevoNivel, this));
                break;

            case 8:
                lista.add(new Estado(swaps(this.estado, 8, 5), nuevoNivel, this));
                lista.add(new Estado(swaps(this.estado, 8, 7), nuevoNivel, this));
                break;
        }

        return lista;
    }

    public void mostrarRuta() {
        if (this.anterior != null) {
            this.anterior.mostrarRuta();
        }

        System.out.println("Tablero:");
        for (int i = 0; i < 9; i++) {
            System.out.print(this.estado.charAt(i) + " ");
            if ((i + 1) % 3 == 0) System.out.println();
        }

        System.out.println();
        System.out.println("Nivel: " + this.profundidad);
        System.out.println("________________________________");
        System.out.println();
    }

    private String swaps(String s, int i, int j) {
        char[] arr = s.toCharArray();
        char tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
        return new String(arr);
    }

    private static String limpiar(String s) {
        if (s == null) return "";

        s = s.replace('0', ' ');

        StringBuilder sb = new StringBuilder();
        for (int k = 0; k < s.length(); k++) {
            char c = s.charAt(k);
            if ((c >= '1' && c <= '8') || c == ' ') sb.append(c);
        }

        if (sb.length() == 9) return sb.toString();
        return s;
    }
}