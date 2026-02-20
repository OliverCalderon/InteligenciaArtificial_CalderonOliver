package diagnostico;

public class Nodo {
    String nombre;
    Nodo izq;
    Nodo der;

    public Nodo(String nombre) {
        this.nombre = nombre;
        this.izq = null;
        this.der = null;
    }
}
