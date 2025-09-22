"""
Tests para el módulo vital_utils
"""
import unittest
import os
import tempfile
import shutil
from unittest.mock import patch
import sys

# Agregar el directorio padre al path para importar los módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.vital_utils import is_nan, key_datetime, find_latest_vital


class TestVitalUtils(unittest.TestCase):
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_is_nan(self):
        """Test para la función is_nan"""
        import math
        
        # Test con valores NaN
        self.assertTrue(is_nan(float('nan')))
        self.assertTrue(is_nan(math.nan))
        
        # Test con valores válidos
        self.assertFalse(is_nan(0))
        self.assertFalse(is_nan(1.5))
        self.assertFalse(is_nan(-10))
        self.assertFalse(is_nan("string"))
        self.assertFalse(is_nan(None))
    
    def test_key_datetime(self):
        """Test para la función key_datetime"""
        # Test con nombres de archivo típicos
        test_cases = [
            ("file_250918_202914.vital", "250918202914"),
            ("VITALLAB_250602_115452.vital", "250602115452"),
            ("test_241225_120000.txt", "241225120000"),
        ]
        
        for filename, expected in test_cases:
            result = key_datetime(filename)
            self.assertEqual(result, expected)
    
    def test_find_latest_vital_no_directory(self):
        """Test cuando el directorio no existe"""
        non_existent_dir = os.path.join(self.temp_dir, "no_existe")
        result = find_latest_vital(non_existent_dir)
        self.assertIsNone(result)
    
    def test_find_latest_vital_no_numeric_folders(self):
        """Test cuando no hay carpetas numéricas"""
        # Crear carpetas no numéricas
        os.makedirs(os.path.join(self.temp_dir, "folder_a"))
        os.makedirs(os.path.join(self.temp_dir, "folder_b"))
        
        result = find_latest_vital(self.temp_dir)
        self.assertIsNone(result)
    
    def test_find_latest_vital_no_vital_files(self):
        """Test cuando hay carpetas numéricas pero sin archivos .vital"""
        # Crear carpeta numérica sin archivos .vital
        folder_path = os.path.join(self.temp_dir, "250918")
        os.makedirs(folder_path)
        
        # Crear archivo que no es .vital
        with open(os.path.join(folder_path, "test.txt"), 'w') as f:
            f.write("test")
        
        result = find_latest_vital(self.temp_dir)
        self.assertIsNone(result)
    
    def test_find_latest_vital_success(self):
        """Test exitoso encontrando el archivo .vital más reciente"""
        # Crear múltiples carpetas numéricas
        folders = ["250917", "250918", "250919"]
        for folder in folders:
            folder_path = os.path.join(self.temp_dir, folder)
            os.makedirs(folder_path)
            
            # Crear archivos .vital con diferentes timestamps
            vital_files = [
                f"file_{folder}_120000.vital",
                f"file_{folder}_130000.vital",
            ]
            
            for vital_file in vital_files:
                with open(os.path.join(folder_path, vital_file), 'w') as f:
                    f.write("mock vital data")
        
        result = find_latest_vital(self.temp_dir)
        
        # Debería encontrar el archivo más reciente en la carpeta más reciente
        expected_path = os.path.join(self.temp_dir, "250919", "file_250919_130000.vital")
        self.assertEqual(result, expected_path)
        self.assertTrue(os.path.exists(result))


if __name__ == '__main__':
    unittest.main()
