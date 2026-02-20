package diagnostico;

public class Arbol {
	private Nodo raiz;

    public Arbol() {
        this.raiz = null;
    }

    public boolean vacio() {
        return raiz == null;
    }

    public Nodo buscarNodo(String nombre) {
        return buscarPreorden(raiz, nombre);
    }

    private Nodo buscarPreorden(Nodo actual, String nombre) {
        if (actual == null) return null;

        if (actual.nombre.equalsIgnoreCase(nombre)) {
            return actual;
        }

        Nodo encontradoIzq = buscarPreorden(actual.izq, nombre);
        if (encontradoIzq != null) return encontradoIzq;

        return buscarPreorden(actual.der, nombre);
    }

    public void insertar(String nombre) {
        raiz = insertarABB(raiz, nombre);
    }

    private Nodo insertarABB(Nodo actual, String nombre) {
        if (actual == null) return new Nodo(nombre);

        int cmp = nombre.compareToIgnoreCase(actual.nombre);

        if (cmp < 0) {
            actual.izq = insertarABB(actual.izq, nombre);
        } else if (cmp > 0) {
            actual.der = insertarABB(actual.der, nombre);
        }

        return actual;
    }
}
