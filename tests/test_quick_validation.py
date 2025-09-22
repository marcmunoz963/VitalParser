"""
Test rápido de validación para verificar que el proyecto funciona
"""
import unittest
import os
import sys

# Agregar el directorio padre al path para importar los módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestQuickValidation(unittest.TestCase):
    """Tests rápidos para validar que el proyecto funciona"""
    
    def test_project_structure(self):
        """Verificar que la estructura del proyecto es correcta"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        # Verificar carpetas principales
        expected_dirs = ['parser', 'models', 'records', 'results']
        for dir_name in expected_dirs:
            dir_path = os.path.join(project_root, dir_name)
            self.assertTrue(os.path.exists(dir_path), f"Directorio {dir_name} no existe")
        
        # Verificar archivos principales
        expected_files = ['vitalParserLearning_GUI.py', 'model.json', 'requirements.txt']
        for file_name in expected_files:
            file_path = os.path.join(project_root, file_name)
            self.assertTrue(os.path.exists(file_path), f"Archivo {file_name} no existe")
    
    def test_imports_work(self):
        """Verificar que los imports principales funcionan"""
        try:
            # Tests de imports básicos
            from parser import vital_utils
            from parser import model_loader
            from parser import vital_processor
            from parser import arr
            
            # Verificar que las funciones principales existen
            self.assertTrue(hasattr(vital_utils, 'find_latest_vital'))
            self.assertTrue(hasattr(vital_utils, 'is_nan'))
            self.assertTrue(hasattr(model_loader, 'load_ml_model'))
            self.assertTrue(hasattr(vital_processor, 'VitalProcessor'))
            self.assertTrue(hasattr(arr, 'interp_undefined'))
            
        except ImportError as e:
            self.fail(f"Error importando módulos: {e}")
    
    def test_records_directory_exists(self):
        """Verificar que la carpeta records existe y tiene contenido"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        records_dir = os.path.join(project_root, 'records')
        
        self.assertTrue(os.path.exists(records_dir), "Carpeta records no existe")
        
        # Verificar que hay al menos una subcarpeta
        subdirs = [d for d in os.listdir(records_dir) 
                  if os.path.isdir(os.path.join(records_dir, d))]
        self.assertGreater(len(subdirs), 0, "No hay subcarpetas en records")
        
        # Verificar que hay al menos un archivo .vital
        vital_files = []
        for root, dirs, files in os.walk(records_dir):
            vital_files.extend([f for f in files if f.endswith('.vital')])
        
        self.assertGreater(len(vital_files), 0, "No hay archivos .vital en records")
    
    def test_model_config_valid(self):
        """Verificar que la configuración de modelos es válida"""
        project_root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(project_root, 'model.json')
        
        self.assertTrue(os.path.exists(config_path), "model.json no existe")
        
        import json
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            self.assertIsInstance(config, list, "Config debe ser una lista")
            self.assertGreater(len(config), 0, "Config no puede estar vacía")
            
            # Verificar estructura de cada modelo
            for model_config in config:
                required_fields = ['name', 'path', 'input_type', 'output_var']
                for field in required_fields:
                    self.assertIn(field, model_config, f"Campo {field} faltante en config")
                
        except json.JSONDecodeError as e:
            self.fail(f"Error parseando model.json: {e}")
        except Exception as e:
            self.fail(f"Error leyendo model.json: {e}")
    
    def test_vital_utils_functions(self):
        """Test rápido de funciones utilitarias"""
        from parser.vital_utils import is_nan, key_datetime
        
        # Test is_nan
        import math
        self.assertTrue(is_nan(float('nan')))
        self.assertFalse(is_nan(5))
        self.assertFalse(is_nan("string"))
        
        # Test key_datetime
        result = key_datetime("test_250918_120000.vital")
        self.assertEqual(result, "250918120000")
    
    def test_arr_basic_functions(self):
        """Test rápido de funciones de procesamiento de arrays"""
        import numpy as np
        from parser import arr
        
        # Test interp_undefined
        test_array = np.array([1, 2, np.nan, 4, 5])
        result = arr.interp_undefined(test_array)
        self.assertFalse(np.isnan(result[2]))  # El NaN debería estar interpolado
        
        # Test exclude_undefined
        result = arr.exclude_undefined(test_array)
        self.assertFalse(np.any(np.isnan(result)))  # No debería haber NaN
        
        # Test resample
        test_data = np.array([1, 2, 3])
        result = arr.resample(test_data, 5)
        self.assertEqual(len(result), 5)
    
    def test_model_loader_basic(self):
        """Test básico del cargador de modelos"""
        from parser.model_loader import PyTorchWrapper
        from unittest.mock import Mock
        
        # Test PyTorchWrapper
        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        
        wrapper = PyTorchWrapper(mock_model)
        self.assertTrue(hasattr(wrapper, 'predict'))
    
    def test_vital_processor_init(self):
        """Test de inicialización del procesador"""
        import tempfile
        from parser.vital_processor import VitalProcessor
        
        with tempfile.TemporaryDirectory() as temp_dir:
            configs = [
                {
                    "name": "Test Model",
                    "input_type": "tabular",
                    "input_vars": ["Demo/ART"],
                    "output_var": "Test_Output",
                    "model": None
                }
            ]
            
            processor = VitalProcessor(configs, temp_dir)
            
            self.assertEqual(processor.model_configs, configs)
            self.assertTrue(os.path.exists(temp_dir))
            self.assertIsNone(processor.latest_df)


if __name__ == '__main__':
    unittest.main()
