"""
Punto de entrada único del Evaluador de ejercicios (lagartijas, sentadillas, curl de bíceps).

    python Evaluador_Ejercicios.py extraer [--ejercicio ...]    -> video -> data/caracteristicas/<ejercicio>.csv
    python Evaluador_Ejercicios.py entrenar [--ejercicio ...]   -> entrena el MLP de cada ejercicio
    python Evaluador_Ejercicios.py evaluar [--ejercicio ...]    -> métricas + matriz de confusión sobre prueba
    python Evaluador_Ejercicios.py tiempo-real --ejercicio ...  -> evaluación en vivo con la webcam
"""

import argparse
import sys


def comando_extraer(args_restantes):
    import extraer_caracteristicas

    sys.argv = [sys.argv[0]] + args_restantes
    extraer_caracteristicas.main()


def comando_entrenar(args_restantes):
    import entrenar

    sys.argv = [sys.argv[0]] + args_restantes
    entrenar.main()


def comando_evaluar(args_restantes):
    import evaluar

    sys.argv = [sys.argv[0]] + args_restantes
    evaluar.main()


def comando_tiempo_real(args_restantes):
    import tiempo_real

    sys.argv = [sys.argv[0]] + args_restantes
    tiempo_real.main()


def main():
    parser = argparse.ArgumentParser(description="Evaluador de ejercicios (lagartijas, sentadillas, curl de bíceps).")
    subparsers = parser.add_subparsers(dest="comando", required=True)
    subparsers.add_parser("extraer", help="Extrae características de repeticiones desde los videos del equipo.")
    subparsers.add_parser("entrenar", help="Entrena el MLP de cada ejercicio.")
    subparsers.add_parser("evaluar", help="Métricas y matriz de confusión sobre la partición de prueba.")
    subparsers.add_parser("tiempo-real", help="Evaluación en vivo con la webcam.")

    args, restantes = parser.parse_known_args()

    if args.comando == "extraer":
        comando_extraer(restantes)
    elif args.comando == "entrenar":
        comando_entrenar(restantes)
    elif args.comando == "evaluar":
        comando_evaluar(restantes)
    elif args.comando == "tiempo-real":
        comando_tiempo_real(restantes)


if __name__ == "__main__":
    main()
