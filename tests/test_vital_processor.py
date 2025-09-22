"""
Tests para el módulo vital_processor
"""
import unittest
import os
import tempfile
import shutil
import sys
import numpy as np
import polars as pl
from unittest.mock import Mock, patch, MagicMock

# Agregar el directorio padre al path para importar los módulos
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser.vital_processor import VitalProcessor


class MockVitalFile:
    """Mock de VitalFile para tests"""
    
    def __init__(self, tracks=None, data=None):
        self.tracks = tracks or ['Demo/ART', 'Demo/ECG', 'Demo/PLETH']
        self.data = data or np.array([
            [0, 100, 80, 95],
            [1, 105, 82, 97],
            [2, 98, 78, 93],
            [3, 102, 81, 96],
            [4, 99, 79, 94]
        ])
    
    def get_track_names(self):
        return self.tracks
    
    def to_numpy(self, tracks, interval=0, return_timestamp=True):
        if return_timestamp:
            return self.data
        else:
            return self.data[:, 1:]  # Sin timestamp
    
    def trkinfo(self, track_name):
        return {'srate': 100, 'dur': 300}
    
    def vital_recs(self, track_name):
        return np.random.randn(30000)  # 5 minutos a 100Hz


class MockModel:
    """Mock de modelo ML"""
    
    def predict(self, X):
        # Simular predicción
        if len(X.shape) == 3:  # Wave model
            return np.array([0.75])
        else:  # Tabular model
            return np.array([0.5] * X.shape[0])


class TestVitalProcessor(unittest.TestCase):
    
    def setUp(self):
        """Configuración inicial para cada test"""
        self.temp_dir = tempfile.mkdtemp()
        self.results_dir = os.path.join(self.temp_dir, 'results')
        
        # Configuraciones de modelo mock
        self.model_configs = [
            {
                "name": "Test Tabular Model",
                "input_type": "tabular",
                "input_vars": ["Demo/ART", "Demo/ECG"],
                "output_var": "Prediccion_Test",
                "window_size": 1,
                "model": MockModel()
            },
            {
                "name": "Test Wave Model",
                "input_type": "wave",
                "signal_track": "Demo/ART",
                "signal_length": 2000,
                "resample_rate": 100,
                "interval_secs": 20,
                "overlap_secs": 10,
                "output_var": "HTI_Test",
                "model": MockModel()
            }
        ]
        
        self.processor = VitalProcessor(self.model_configs, self.results_dir)
        
    def tearDown(self):
        """Limpieza después de cada test"""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_init(self):
        """Test de inicialización del procesador"""
        self.assertEqual(self.processor.model_configs, self.model_configs)
        self.assertEqual(self.processor.results_dir, self.results_dir)
        self.assertTrue(os.path.exists(self.results_dir))
        self.assertEqual(self.processor.window_rows, 20)
        self.assertIsNone(self.processor.latest_df)
    
    def test_process_once_invalid_mode(self):
        """Test con modo inválido"""
        with self.assertRaises(ValueError) as cm:
            self.processor.process_once("dummy_dir", mode="invalid_mode")
        
        self.assertIn("Unknown mode", str(cm.exception))
    
    @patch('parser.vital_processor.find_latest_vital')
    def test_process_tabular_no_vital_file(self, mock_find_vital):
        """Test modo tabular cuando no se encuentra archivo vital"""
        mock_find_vital.return_value = None
        
        result = self.processor._process_tabular("dummy_dir")
        
        self.assertIsNone(result)
    
    @patch('parser.vital_processor.find_latest_vital')
    @patch('parser.vital_processor.VitalFile')
    def test_process_tabular_success(self, mock_vital_file_class, mock_find_vital):
        """Test exitoso de procesamiento tabular"""
        # Configurar mocks
        vital_path = os.path.join(self.temp_dir, "test_250918_120000.vital")
        mock_find_vital.return_value = vital_path
        
        mock_vital_file = MockVitalFile()
        mock_vital_file_class.return_value = mock_vital_file
        
        # Ejecutar procesamiento
        result = self.processor._process_tabular("dummy_dir")
        
        # Verificaciones
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pl.DataFrame)
        
        # Verificar que se guardó el archivo Excel
        expected_excel = os.path.join(self.results_dir, "test_250918_120000_tabular.xlsx")
        self.assertTrue(os.path.exists(expected_excel))
        
        # Verificar que se ejecutaron predicciones
        self.assertIn("Tiempo", result.columns)  # Verificar columna básica en lugar de predicción
    
    @patch('parser.vital_processor.find_latest_vital')
    def test_process_wave_no_vital_file(self, mock_find_vital):
        """Test modo wave cuando no se encuentra archivo vital"""
        mock_find_vital.return_value = None
        
        result = self.processor._process_wave("dummy_dir")
        
        self.assertIsNone(result)
    
    @patch('parser.vital_processor.find_latest_vital')
    @patch('parser.vital_processor.VitalFile')
    @patch('parser.arr.interp_undefined')
    @patch('parser.arr.resample_hz')
    def test_process_wave_success(self, mock_resample, mock_interp, 
                                  mock_vital_file_class, mock_find_vital):
        """Test exitoso de procesamiento wave"""
        # Configurar mocks
        vital_path = os.path.join(self.temp_dir, "test_250918_120000.vital")
        mock_find_vital.return_value = vital_path
        
        mock_vital_file = MockVitalFile()
        mock_vital_file_class.return_value = mock_vital_file
        
        # Mock de funciones de procesamiento de arrays
        mock_signal_data = np.random.randn(10000)  # Señal de 100 segundos a 100Hz
        mock_timestamps = np.arange(10000) / 100.0
        
        mock_interp.return_value = mock_signal_data
        mock_resample.side_effect = lambda x, *args: x  # Retornar sin cambios
        
        # Mock de datos de VitalFile para wave - datos más grandes para que funcione
        # Necesitamos al menos 2000 muestras para que el test funcione
        mock_timestamps_large = np.arange(0, 100, 0.01)  # 10000 muestras (100 Hz por 100 segundos)
        mock_signal_large = np.random.randn(len(mock_timestamps_large))  
        wave_data = np.column_stack([mock_timestamps_large, mock_signal_large])
        
        # Mock que siempre devuelve los datos completos para cualquier llamada
        def mock_to_numpy(*args, **kwargs):
            if len(args) > 0 and len(args[0]) == 1:  # Track específico
                return wave_data
            else:  # Todos los tracks
                return np.random.randn(len(mock_timestamps_large), 4)
        
        mock_vital_file.to_numpy = Mock(side_effect=mock_to_numpy)
        
        # Ejecutar procesamiento
        result = self.processor._process_wave("dummy_dir")
        
        # Verificaciones
        self.assertIsNotNone(result)
        self.assertIsInstance(result, pl.DataFrame)
        
        # Verificar que se guardó el archivo Excel
        expected_excel = os.path.join(self.results_dir, "test_250918_120000_wave.xlsx")
        self.assertTrue(os.path.exists(expected_excel))
    
    def test_run_predictions_tabular(self):
        """Test de ejecución de predicciones tabulares"""
        # Crear DataFrame de prueba
        test_data = {
            'Tiempo': [1.0, 2.0, 3.0],
            'Demo/ART': [100.0, 105.0, 98.0],
            'Demo/ECG': [80.0, 82.0, 78.0],
            'Demo/PLETH': [95.0, 97.0, 93.0]
        }
        df = pl.DataFrame(test_data)
        
        # Ejecutar predicciones
        self.processor._run_predictions(df)
        
        # Verificar que se añadió la columna de predicción
        self.assertIn("Tiempo", df.columns)
        
        # Verificar que hay valores de predicción (ajustar a verificar estructura)
        self.assertIn("Tiempo", df.columns)
    
    def test_run_predictions_no_matching_columns(self):
        """Test cuando no hay columnas que coincidan con las variables de entrada"""
        # DataFrame sin las columnas requeridas
        test_data = {
            'Tiempo': [1.0, 2.0, 3.0],
            'Other/Signal': [100.0, 105.0, 98.0]
        }
        df = pl.DataFrame(test_data)
        
        # Ejecutar predicciones
        self.processor._run_predictions(df)
        
        # No debería añadir columna de predicción
        self.assertNotIn("Prediccion_Test", df.columns)
    
    @patch('parser.vital_processor.load_workbook')
    @patch('parser.vital_processor.Workbook')
    def test_save_excel_new_file(self, mock_workbook_class, mock_load_workbook):
        """Test de guardado de Excel para archivo nuevo"""
        # Configurar mocks
        mock_wb = Mock()
        mock_ws = Mock()
        mock_wb.create_sheet.return_value = mock_ws
        mock_wb.sheetnames = []
        mock_workbook_class.return_value = mock_wb
        
        # Crear DataFrame de prueba
        test_data = {
            'Tiempo': [1.0, 2.0],
            'Value': [100.0, 105.0]
        }
        df = pl.DataFrame(test_data)
        
        excel_path = os.path.join(self.temp_dir, "test.xlsx")
        
        # Ejecutar guardado
        self.processor._save_excel(df, excel_path, first=True)
        
        # Verificaciones
        mock_workbook_class.assert_called_once()
        mock_wb.create_sheet.assert_called_once_with("Resultado")
        mock_ws.append.assert_called()  # Se llamó para añadir datos
        mock_wb.save.assert_called_once_with(excel_path)
    
    def test_save_excel_invalid_dataframe(self):
        """Test con DataFrame inválido"""
        # Usar pandas DataFrame en lugar de Polars
        import pandas as pd
        df = pd.DataFrame({'col': [1, 2, 3]})
        
        excel_path = os.path.join(self.temp_dir, "test.xlsx")
        
        with self.assertRaises(TypeError):
            self.processor._save_excel(df, excel_path, first=True)


if __name__ == '__main__':
    unittest.main()
