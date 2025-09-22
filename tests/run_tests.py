#!/usr/bin/env python3
"""
Script principal para ejecutar todos los tests de VitalParser
"""
import unittest
import sys
import os
import time
from io import StringIO

# Agregar el directorio del proyecto al path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ColoredTextTestResult(unittest.TextTestResult):
    """Resultado de tests con colores para mejor legibilidad"""
    
    def __init__(self, stream, descriptions, verbosity):
        super().__init__(stream, descriptions, verbosity)
        self.success_count = 0
        self.verbosity = verbosity
        
    def addSuccess(self, test):
        super().addSuccess(test)
        self.success_count += 1
        if self.verbosity > 1:
            self.stream.write(f"✅ {test._testMethodName}")
            self.stream.write(" ... ")
            self.stream.writeln("OK")
    
    def addError(self, test, err):
        super().addError(test, err)
        if self.verbosity > 1:
            self.stream.write(f"❌ {test._testMethodName}")
            self.stream.write(" ... ")
            self.stream.writeln("ERROR")
    
    def addFailure(self, test, err):
        super().addFailure(test, err)
        if self.verbosity > 1:
            self.stream.write(f"❌ {test._testMethodName}")
            self.stream.write(" ... ")
            self.stream.writeln("FAIL")
    
    def addSkip(self, test, reason):
        super().addSkip(test, reason)
        if self.verbosity > 1:
            self.stream.write(f"⏭️  {test._testMethodName}")
            self.stream.write(" ... ")
            self.stream.writeln(f"SKIPPED ({reason})")


class VitalParserTestRunner:
    """Runner principal para tests de VitalParser"""
    
    def __init__(self):
        self.test_modules = [
            'tests.test_vital_utils',
            'tests.test_model_loader', 
            'tests.test_vital_processor',
            'tests.test_arr',
            'tests.test_integration'
        ]
        
    def run_single_module(self, module_name, verbosity=2):
        """Ejecutar tests de un módulo específico"""
        print(f"\n{'='*60}")
        print(f"🧪 EJECUTANDO TESTS: {module_name}")
        print(f"{'='*60}")
        
        try:
            # Cargar tests del módulo
            loader = unittest.TestLoader()
            suite = loader.loadTestsFromName(module_name)
            
            # Ejecutar tests
            runner = unittest.TextTestRunner(
                verbosity=verbosity,
                resultclass=ColoredTextTestResult,
                stream=sys.stdout
            )
            
            start_time = time.time()
            result = runner.run(suite)
            end_time = time.time()
            
            # Estadísticas del módulo
            total_tests = result.testsRun
            errors = len(result.errors)
            failures = len(result.failures)
            skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
            success = getattr(result, 'success_count', total_tests - errors - failures)
            
            print(f"\n📊 ESTADÍSTICAS DEL MÓDULO:")
            print(f"   ✅ Exitosos: {success}")
            print(f"   ❌ Errores: {errors}")
            print(f"   ❌ Fallos: {failures}")
            print(f"   ⏭️  Omitidos: {skipped}")
            print(f"   ⏱️  Tiempo: {end_time - start_time:.2f}s")
            
            return result
            
        except Exception as e:
            print(f"❌ Error cargando módulo {module_name}: {e}")
            return None
    
    def run_all_tests(self, verbosity=2, stop_on_failure=False):
        """Ejecutar todos los tests"""
        print("🚀 INICIANDO SUITE COMPLETA DE TESTS VITALPARSER")
        print("="*80)
        
        total_start_time = time.time()
        all_results = []
        
        # Estadísticas globales
        total_tests = 0
        total_errors = 0
        total_failures = 0
        total_skipped = 0
        total_success = 0
        
        # Ejecutar cada módulo
        for module_name in self.test_modules:
            result = self.run_single_module(module_name, verbosity)
            
            if result is None:
                print(f"⚠️  Saltando módulo {module_name} por error de carga")
                continue
                
            all_results.append((module_name, result))
            
            # Acumular estadísticas
            total_tests += result.testsRun
            total_errors += len(result.errors)
            total_failures += len(result.failures)
            total_skipped += len(result.skipped) if hasattr(result, 'skipped') else 0
            total_success += getattr(result, 'success_count', result.testsRun - len(result.errors) - len(result.failures))
            
            # Parar en caso de fallo si está configurado
            if stop_on_failure and (len(result.errors) > 0 or len(result.failures) > 0):
                print(f"\n🛑 DETENIENDO TESTS POR FALLO EN {module_name}")
                break
        
        total_end_time = time.time()
        total_time = total_end_time - total_start_time
        
        # Reporte final
        self.print_final_report(all_results, total_tests, total_success, 
                              total_errors, total_failures, total_skipped, total_time)
        
        return all_results
    
    def print_final_report(self, all_results, total_tests, total_success, 
                          total_errors, total_failures, total_skipped, total_time):
        """Imprimir reporte final detallado"""
        print("\n" + "="*80)
        print("📋 REPORTE FINAL DE TESTS")
        print("="*80)
        
        # Resumen por módulo
        print("\n📊 RESUMEN POR MÓDULO:")
        print("-" * 80)
        print(f"{'Módulo':<30} {'Tests':<8} {'✅ OK':<8} {'❌ Error':<8} {'❌ Fallo':<8} {'⏭️ Skip':<8}")
        print("-" * 80)
        
        for module_name, result in all_results:
            module_short = module_name.split('.')[-1]
            tests = result.testsRun
            errors = len(result.errors)
            failures = len(result.failures)
            skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
            success = getattr(result, 'success_count', tests - errors - failures)
            
            print(f"{module_short:<30} {tests:<8} {success:<8} {errors:<8} {failures:<8} {skipped:<8}")
        
        print("-" * 80)
        print(f"{'TOTAL':<30} {total_tests:<8} {total_success:<8} {total_errors:<8} {total_failures:<8} {total_skipped:<8}")
        
        # Estadísticas globales
        print(f"\n🎯 ESTADÍSTICAS GLOBALES:")
        print(f"   📝 Total de tests ejecutados: {total_tests}")
        print(f"   ✅ Tests exitosos: {total_success}")
        print(f"   ❌ Tests con errores: {total_errors}")
        print(f"   ❌ Tests fallidos: {total_failures}")
        print(f"   ⏭️  Tests omitidos: {total_skipped}")
        print(f"   ⏱️  Tiempo total: {total_time:.2f} segundos")
        
        # Porcentaje de éxito
        if total_tests > 0:
            success_rate = (total_success / total_tests) * 100
            print(f"   🎯 Tasa de éxito: {success_rate:.1f}%")
            
            if success_rate >= 90:
                print("   🎉 ¡EXCELENTE! Suite de tests muy estable")
            elif success_rate >= 75:
                print("   👍 BUENO. Suite de tests estable")
            elif success_rate >= 50:
                print("   ⚠️  REGULAR. Algunos tests necesitan atención")
            else:
                print("   🚨 CRÍTICO. Muchos tests fallando")
        
        # Detalles de errores si los hay
        if total_errors > 0 or total_failures > 0:
            print(f"\n🔍 DETALLES DE FALLOS:")
            print("-" * 60)
            
            for module_name, result in all_results:
                if len(result.errors) > 0 or len(result.failures) > 0:
                    print(f"\n📁 {module_name}:")
                    
                    for test, error in result.errors:
                        print(f"   ❌ ERROR en {test}: {error.split(chr(10))[0]}")
                    
                    for test, failure in result.failures:
                        print(f"   ❌ FALLO en {test}: {failure.split(chr(10))[0]}")
        
        print("\n" + "="*80)
        
        # Resultado final
        if total_errors == 0 and total_failures == 0:
            print("🎉 ¡TODOS LOS TESTS PASARON EXITOSAMENTE!")
        else:
            print("⚠️  ALGUNOS TESTS FALLARON. REVISAR DETALLES ARRIBA.")
        
        print("="*80)
    
    def run_specific_test(self, test_pattern, verbosity=2):
        """Ejecutar tests específicos basados en un patrón"""
        print(f"🎯 EJECUTANDO TESTS ESPECÍFICOS: {test_pattern}")
        
        loader = unittest.TestLoader()
        suite = loader.loadTestsFromName(test_pattern)
        
        runner = unittest.TextTestRunner(
            verbosity=verbosity,
            resultclass=ColoredTextTestResult
        )
        
        return runner.run(suite)


def main():
    """Función principal"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Ejecutar tests de VitalParser')
    parser.add_argument('--module', '-m', help='Ejecutar tests de un módulo específico')
    parser.add_argument('--test', '-t', help='Ejecutar un test específico')
    parser.add_argument('--verbose', '-v', action='store_true', help='Salida verbose')
    parser.add_argument('--quiet', '-q', action='store_true', help='Salida minimal')
    parser.add_argument('--stop-on-failure', '-s', action='store_true', 
                       help='Parar en el primer fallo')
    
    args = parser.parse_args()
    
    # Configurar verbosidad
    if args.quiet:
        verbosity = 0
    elif args.verbose:
        verbosity = 2
    else:
        verbosity = 1
    
    # Crear runner
    runner = VitalParserTestRunner()
    
    try:
        if args.test:
            # Ejecutar test específico
            runner.run_specific_test(args.test, verbosity)
        elif args.module:
            # Ejecutar módulo específico
            runner.run_single_module(f'tests.{args.module}', verbosity)
        else:
            # Ejecutar todos los tests
            runner.run_all_tests(verbosity, args.stop_on_failure)
            
    except KeyboardInterrupt:
        print("\n🛑 Tests interrumpidos por el usuario")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Error inesperado: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
