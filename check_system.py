#!/usr/bin/env python3
"""
VitalParser - Script de Verificación del Sistema
Comprueba dependencias, ejecuta tests y valida la instalación
"""

import sys
import os
import subprocess
import importlib.util
import time
import json
from pathlib import Path

class Colors:
    """Colores para output en terminal"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    END = '\033[0m'

def print_header(title):
    """Imprime un encabezado con estilo"""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'='*60}")
    print(f"    {title}")
    print(f"{'='*60}{Colors.END}")

def print_success(message):
    """Imprime mensaje de éxito"""
    print(f"{Colors.GREEN}✅ {message}{Colors.END}")

def print_error(message):
    """Imprime mensaje de error"""
    print(f"{Colors.RED}❌ {message}{Colors.END}")

def print_warning(message):
    """Imprime mensaje de advertencia"""
    print(f"{Colors.YELLOW}⚠️  {message}{Colors.END}")

def print_info(message):
    """Imprime mensaje informativo"""
    print(f"{Colors.BLUE}ℹ️  {message}{Colors.END}")

def check_python_version():
    """Verifica la versión de Python"""
    print_header("VERIFICACIÓN DE PYTHON")
    
    version = sys.version_info
    print(f"Versión detectada: {version.major}.{version.minor}.{version.micro}")
    
    if version >= (3, 8):
        print_success(f"Python {version.major}.{version.minor}.{version.micro} es compatible")
        return True
    else:
        print_error(f"Se requiere Python 3.8+ (actual: {version.major}.{version.minor})")
        return False

def check_dependencies():
    """Verifica todas las dependencias"""
    print_header("VERIFICACIÓN DE DEPENDENCIAS")
    
    # Dependencias principales
    main_deps = {
        'vitaldb': 'Procesamiento de archivos .vital',
        'polars': 'Manipulación de DataFrames',
        'sklearn': 'Machine Learning (scikit-learn)',
        'tensorflow': 'Redes neuronales',
        'numpy': 'Computación numérica',
        'scipy': 'Procesamiento de señales',
        'openpyxl': 'Exportación Excel',
        'tkinter': 'Interfaz gráfica'
    }
    
    # Dependencias opcionales
    optional_deps = {
        'torch': 'PyTorch (opcional)',
        'pandas': 'Pandas (opcional)',
        'psutil': 'Monitoreo del sistema (opcional)',
        'joblib': 'Serialización (opcional)',
        'PyPDF2': 'Procesamiento PDF (opcional)'
    }
    
    missing_main = []
    missing_optional = []
    
    # Verificar dependencias principales
    print(f"\n{Colors.BOLD}Dependencias principales:{Colors.END}")
    for module, description in main_deps.items():
        try:
            if module == 'sklearn':
                importlib.import_module('sklearn')
            elif module == 'tkinter':
                importlib.import_module('tkinter')
            else:
                importlib.import_module(module)
            print_success(f"{module:<12} - {description}")
        except ImportError:
            print_error(f"{module:<12} - {description}")
            missing_main.append(module)
    
    # Verificar dependencias opcionales
    print(f"\n{Colors.BOLD}Dependencias opcionales:{Colors.END}")
    for module, description in optional_deps.items():
        try:
            importlib.import_module(module)
            print_success(f"{module:<12} - {description}")
        except ImportError:
            print_warning(f"{module:<12} - {description}")
            missing_optional.append(module)
    
    # Resumen
    print(f"\n{Colors.BOLD}Resumen:{Colors.END}")
    if not missing_main:
        print_success("Todas las dependencias principales están instaladas")
        main_ok = True
    else:
        print_error(f"Faltan {len(missing_main)} dependencias principales: {', '.join(missing_main)}")
        main_ok = False
    
    if missing_optional:
        print_warning(f"Dependencias opcionales faltantes: {', '.join(missing_optional)}")
    
    return main_ok, missing_main, missing_optional

def check_project_structure():
    """Verifica la estructura del proyecto"""
    print_header("VERIFICACIÓN DE ESTRUCTURA DEL PROYECTO")
    
    required_files = [
        'vitalParserLearning_GUI.py',
        'model.json',
        'requirements.txt',
        'README.md'
    ]
    
    required_dirs = [
        'parser',
        'models',
        'tests',
        'records',
        'results'
    ]
    
    missing_files = []
    missing_dirs = []
    
    # Verificar archivos
    print(f"\n{Colors.BOLD}Archivos principales:{Colors.END}")
    for file in required_files:
        if os.path.exists(file):
            print_success(f"{file}")
        else:
            print_error(f"{file}")
            missing_files.append(file)
    
    # Verificar directorios
    print(f"\n{Colors.BOLD}Directorios principales:{Colors.END}")
    for dir_name in required_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            print_success(f"{dir_name}/")
        else:
            print_error(f"{dir_name}/")
            missing_dirs.append(dir_name)
    
    # Verificar archivos específicos en directorios
    print(f"\n{Colors.BOLD}Archivos en subdirectorios:{Colors.END}")
    
    parser_files = ['vital_processor.py', 'model_loader.py', 'gui.py']
    for file in parser_files:
        path = os.path.join('parser', file)
        if os.path.exists(path):
            print_success(f"parser/{file}")
        else:
            print_error(f"parser/{file}")
    
    model_files = ['pleth_bp_predictor.py']
    for file in model_files:
        path = os.path.join('models', file)
        if os.path.exists(path):
            print_success(f"models/{file}")
        else:
            print_error(f"models/{file}")
    
    test_files = ['test_pleth_bp.py', 'test_batch_processing.py']
    for file in test_files:
        path = os.path.join('tests', file)
        if os.path.exists(path):
            print_success(f"tests/{file}")
        else:
            print_error(f"tests/{file}")
    
    return len(missing_files) == 0 and len(missing_dirs) == 0

def check_model_configuration():
    """Verifica la configuración de modelos"""
    print_header("VERIFICACIÓN DE CONFIGURACIÓN DE MODELOS")
    
    try:
        with open('model.json', 'r') as f:
            configs = json.load(f)
        
        print_success("Archivo model.json cargado correctamente")
        
        # Verificar estructura
        if not isinstance(configs, list):
            print_error("model.json debe contener una lista de configuraciones")
            return False
        
        print_info(f"Se encontraron {len(configs)} configuraciones de modelos")
        
        # Verificar cada configuración
        for i, config in enumerate(configs):
            print(f"\n{Colors.BOLD}Modelo {i+1}:{Colors.END}")
            
            required_fields = ['name', 'path', 'input_type']
            for field in required_fields:
                if field in config:
                    print_success(f"  {field}: {config[field]}")
                else:
                    print_error(f"  {field}: FALTANTE")
            
            # Verificar archivo del modelo
            if 'path' in config:
                model_path = config['path']
                if os.path.exists(model_path):
                    print_success(f"  Archivo modelo: {model_path}")
                else:
                    print_error(f"  Archivo modelo: {model_path} NO ENCONTRADO")
        
        return True
        
    except FileNotFoundError:
        print_error("Archivo model.json no encontrado")
        return False
    except json.JSONDecodeError as e:
        print_error(f"Error al parsear model.json: {e}")
        return False

def run_tests():
    """Ejecuta todos los tests del sistema"""
    print_header("EJECUCIÓN DE TESTS")
    
    # Buscar todos los archivos de test en el directorio tests/
    test_files = []
    tests_dir = 'tests'
    if os.path.exists(tests_dir):
        for file in os.listdir(tests_dir):
            if file.startswith('test_') and file.endswith('.py'):
                test_files.append(os.path.join(tests_dir, file))
    
    # Ordenar para ejecución consistente
    test_files.sort()
    
    print(f"[INFO] Encontrados {len(test_files)} archivos de test:")
    for test_file in test_files:
        print(f"  - {test_file}")
    print()
    
    results = {}
    
    for test_file in test_files:
        if not os.path.exists(test_file):
            print_warning(f"Test no encontrado: {test_file}")
            continue
        
        print(f"\n{Colors.BOLD}Ejecutando: {test_file}{Colors.END}")
        print(f"{Colors.CYAN}{'─'*50}{Colors.END}")
        
        try:
            start_time = time.time()
            # Timeout más generoso para tests de procesamiento
            if 'performance' in test_file or 'wave' in test_file or 'mock_data' in test_file:
                timeout = 900  # 15 minutos para tests pesados
            elif 'batch_processing' in test_file or 'pleth_bp' in test_file:
                timeout = 600  # 10 minutos para tests de procesamiento
            else:
                timeout = 300  # 5 minutos para tests normales
            result = subprocess.run([sys.executable, test_file], 
                                  capture_output=True, text=True, timeout=timeout)
            end_time = time.time()
            
            if result.returncode == 0:
                print_success(f"Test completado en {end_time - start_time:.1f}s")
                results[test_file] = {'status': 'PASS', 'time': end_time - start_time}
            else:
                print_error(f"Test falló (código: {result.returncode})")
                print(f"{Colors.RED}Error:{Colors.END}")
                print(result.stderr)
                results[test_file] = {'status': 'FAIL', 'error': result.stderr}
                
        except subprocess.TimeoutExpired:
            timeout_minutes = timeout // 60
            print_error(f"Test excedió el tiempo límite ({timeout_minutes} minutos)")
            results[test_file] = {'status': 'TIMEOUT', 'timeout': timeout}
        except Exception as e:
            print_error(f"Error ejecutando test: {e}")
            results[test_file] = {'status': 'ERROR', 'error': str(e)}
    
    # Resumen de tests
    print(f"\n{Colors.BOLD}Resumen de Tests:{Colors.END}")
    passed = sum(1 for r in results.values() if r['status'] == 'PASS')
    failed = sum(1 for r in results.values() if r['status'] == 'FAIL')
    timeout = sum(1 for r in results.values() if r['status'] == 'TIMEOUT')
    error = sum(1 for r in results.values() if r['status'] == 'ERROR')
    total = len(results)
    
    print(f"[INFO] Total de tests: {total}")
    print_success(f"Exitosos: {passed}")
    if failed > 0:
        print_error(f"Fallidos: {failed}")
    if timeout > 0:
        print_warning(f"Timeout: {timeout}")
    if error > 0:
        print_error(f"Errores: {error}")
    
    if passed == total:
        print_success(f"🎉 Todos los tests pasaron ({passed}/{total})")
    elif passed > 0:
        print_warning(f"⚠️  {passed}/{total} tests pasaron")
    else:
        print_error(f"❌ Ningún test pasó ({passed}/{total})")
    
    return results

def check_data_files():
    """Verifica la presencia de archivos de datos"""
    print_header("VERIFICACIÓN DE ARCHIVOS DE DATOS")
    
    records_dir = 'records'
    if not os.path.exists(records_dir):
        print_warning("Directorio 'records' no encontrado")
        return False
    
    # Buscar archivos .vital
    vital_files = []
    for root, dirs, files in os.walk(records_dir):
        for file in files:
            if file.endswith('.vital'):
                vital_files.append(os.path.join(root, file))
    
    if vital_files:
        print_success(f"Se encontraron {len(vital_files)} archivos .vital")
        for file in vital_files[:5]:  # Mostrar solo los primeros 5
            print_info(f"  {file}")
        if len(vital_files) > 5:
            print_info(f"  ... y {len(vital_files) - 5} más")
    else:
        print_warning("No se encontraron archivos .vital en records/")
    
    return len(vital_files) > 0

def generate_report(results):
    """Genera un reporte final"""
    print_header("REPORTE FINAL")
    
    # Resumen general
    python_ok = results.get('python', False)
    deps_ok = results.get('dependencies', False)
    structure_ok = results.get('structure', False)
    config_ok = results.get('configuration', False)
    data_ok = results.get('data', False)
    tests_results = results.get('tests', {})
    
    print(f"\n{Colors.BOLD}Estado del Sistema:{Colors.END}")
    print(f"Python:           {'✅ OK' if python_ok else '❌ ERROR'}")
    print(f"Dependencias:     {'✅ OK' if deps_ok else '❌ ERROR'}")
    print(f"Estructura:       {'✅ OK' if structure_ok else '❌ ERROR'}")
    print(f"Configuración:   {'✅ OK' if config_ok else '❌ ERROR'}")
    print(f"Datos:           {'✅ OK' if data_ok else '⚠️  ADVERTENCIA'}")
    
    # Tests
    if tests_results:
        passed_tests = sum(1 for r in tests_results.values() if r['status'] == 'PASS')
        total_tests = len(tests_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.8:  # 80% o más de éxito
            print(f"Tests:           ✅ OK ({passed_tests}/{total_tests})")
        elif success_rate >= 0.6:  # 60-79% de éxito
            print(f"Tests:           ⚠️  ADVERTENCIA ({passed_tests}/{total_tests})")
        else:  # Menos de 60% de éxito
            print(f"Tests:           ❌ ERROR ({passed_tests}/{total_tests})")
    
    # Recomendaciones
    print(f"\n{Colors.BOLD}Recomendaciones:{Colors.END}")
    
    if not python_ok:
        print_error("Actualizar Python a versión 3.8+")
    
    if not deps_ok:
        print_error("Instalar dependencias faltantes: pip install -r requirements.txt")
    
    if not structure_ok:
        print_error("Verificar estructura del proyecto")
    
    if not config_ok:
        print_error("Revisar configuración en model.json")
    
    if not data_ok:
        print_warning("Agregar archivos .vital en directorio records/")
    
    # Estado final
    all_critical_ok = python_ok and deps_ok and structure_ok and config_ok
    
    # Considerar tests en el estado final
    if tests_results:
        passed_tests = sum(1 for r in tests_results.values() if r['status'] == 'PASS')
        total_tests = len(tests_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        tests_ok = success_rate >= 0.8  # 80% o más de éxito
    else:
        tests_ok = True  # Si no hay tests, no afecta el estado
    
    print(f"\n{Colors.BOLD}Estado Final:{Colors.END}")
    if all_critical_ok and tests_ok:
        print_success("🎉 Sistema VitalParser listo para usar!")
        print_info("Ejecuta: python vitalParserLearning_GUI.py")
    elif all_critical_ok and not tests_ok:
        print_warning("⚠️  Sistema funcional pero algunos tests fallan")
        print_info("Ejecuta: python vitalParserLearning_GUI.py")
    else:
        print_error("🚨 Sistema requiere atención antes de usar")
    
    return all_critical_ok

def main():
    """Función principal"""
    print(f"{Colors.PURPLE}{Colors.BOLD}")
    print("="*60)
    print("    VitalParser - Verificación del Sistema")
    print("    Comprobación completa de dependencias y tests")
    print("="*60)
    print(f"{Colors.END}")
    
    results = {}
    
    # Ejecutar todas las verificaciones
    results['python'] = check_python_version()
    deps_ok, missing_main, missing_optional = check_dependencies()
    results['dependencies'] = deps_ok
    results['structure'] = check_project_structure()
    results['configuration'] = check_model_configuration()
    results['data'] = check_data_files()
    
    # Ejecutar tests solo si las dependencias están OK
    if deps_ok:
        results['tests'] = run_tests()
    else:
        print_warning("Saltando tests debido a dependencias faltantes")
        results['tests'] = {}
    
    # Generar reporte final
    system_ready = generate_report(results)
    
    return 0 if system_ready else 1

if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
