package diagnostico;

public class Main {
	public static void main(String[] args) {
		
        Arbol arbol = new Arbol();

        System.out.println("¿El árbol está vacío? " + arbol.vacio()); //verificar que sirve arbol vacio

        arbol.insertar("Carlos");
        arbol.insertar("Ana");
        arbol.insertar("Pedro");
        arbol.insertar("Luis");
        arbol.insertar("Sofia");

        System.out.println("¿El árbol está vacío? " + arbol.vacio()); //verificar que sirve arbol lleno

        Nodo encontrado = arbol.buscarNodo("Pedro");
        if (encontrado != null) {
            System.out.println("Nodo encontrado: " + encontrado.nombre);
        } else {
            System.out.println("Nodo no encontrado");
        }

        Nodo noExiste = arbol.buscarNodo("Juan");
        if (noExiste != null) {
            System.out.println("Nodo encontrado: " + noExiste.nombre);
        } else {
            System.out.println("Nodo Juan no existe en el árbol");
        }
	}
}