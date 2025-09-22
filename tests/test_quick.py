#!/usr/bin/env python3
"""
Script para ejecutar un test rápido y verificar que todo funciona
"""
import sys
import os
import unittest

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_test():
    """Ejecutar test rápido de validación"""
    print("[INFO] Ejecutando test rápido de validación...")
    print("="*50)
    
    # Importar y ejecutar solo el test de validación rápida
    from test_quick_validation import TestQuickValidation
    
    # Crear suite con solo los tests rápidos
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestQuickValidation)
    
    # Ejecutar con verbosidad
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Mostrar resultado
    print("\n" + "="*50)
    if result.wasSuccessful():
        print("[OK] ¡TEST RÁPIDO EXITOSO! El proyecto funciona correctamente.")
        print(f"[INFO] Tests ejecutados: {result.testsRun}")
        print("[OK] Todos los componentes básicos están funcionando.")
    else:
        print("[ERROR] ALGUNOS TESTS FALLARON")
        print(f"[INFO] Tests ejecutados: {result.testsRun}")
        print(f"[ERROR] Errores: {len(result.errors)}")
        print(f"[ERROR] Fallos: {len(result.failures)}")
        
        if result.errors:
            print("\n[ERROR] ERRORES:")
            for test, error in result.errors:
                print(f"   • {test}: {error.split(chr(10))[0]}")
        
        if result.failures:
            print("\n[ERROR] FALLOS:")
            for test, failure in result.failures:
                print(f"   • {test}: {failure.split(chr(10))[0]}")
    
    print("="*50)
    return result.wasSuccessful()

if __name__ == '__main__':
    success = quick_test()
    sys.exit(0 if success else 1)
