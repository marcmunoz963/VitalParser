"""
Generador de datos mock para tests
"""
import os
import numpy as np
import tempfile
import shutil
from datetime import datetime, timedelta


class MockDataGenerator:
    """Generador de datos de prueba para VitalParser"""
    
    def __init__(self, base_dir=None):
        self.base_dir = base_dir or tempfile.mkdtemp()
        self.srate = 100  # Frecuencia de muestreo
        
    def create_mock_records_structure(self, num_days=3):
        """Crear estructura de carpetas de records mock"""
        records_dir = os.path.join(self.base_dir, 'records')
        
        for i in range(num_days):
            date = datetime.now() - timedelta(days=i)
            date_str = date.strftime('%y%m%d')
            
            day_dir = os.path.join(records_dir, date_str)
            os.makedirs(day_dir, exist_ok=True)
            
            # Crear varios archivos vital por día
            for hour in [10, 14, 18]:
                timestamp = f"{date_str}_{hour:02d}0000"
                vital_filename = f"VITALLAB_{timestamp}.vital"
                vital_path = os.path.join(day_dir, vital_filename)
                
                self._create_mock_vital_file(vital_path, duration_minutes=30)
        
        return records_dir
    
    def _create_mock_vital_file(self, file_path, duration_minutes=10):
        """Crear archivo .vital mock con datos binarios simulados"""
        # Crear datos binarios mock que simulan un archivo vital
        # En realidad, los archivos .vital tienen un formato específico,
        # pero para tests podemos usar datos mock
        
        duration_seconds = duration_minutes * 60
        samples = duration_seconds * self.srate
        
        # Generar señales sintéticas
        t = np.linspace(0, duration_seconds, samples)
        
        # Señales médicas simuladas
        art_signal = self._generate_arterial_pressure(t)
        ecg_signal = self._generate_ecg_signal(t)
        pleth_signal = self._generate_pleth_signal(t)
        
        # Crear estructura de datos mock
        mock_data = {
            'header': b'VITAL_FILE_MOCK_v1.0',
            'duration': duration_seconds,
            'srate': self.srate,
            'signals': {
                'Demo/ART': art_signal,
                'Demo/ECG': ecg_signal,
                'Demo/PLETH': pleth_signal
            }
        }
        
        # Guardar como archivo binario simple
        # (En un caso real, usaríamos el formato específico de VitalDB)
        with open(file_path, 'wb') as f:
            # Header simple
            f.write(mock_data['header'])
            f.write(duration_seconds.to_bytes(4, 'little'))
            f.write(self.srate.to_bytes(4, 'little'))
            
            # Datos de señales (simplificado)
            for signal_name, signal_data in mock_data['signals'].items():
                signal_bytes = signal_data.astype(np.float32).tobytes()
                f.write(len(signal_bytes).to_bytes(4, 'little'))
                f.write(signal_bytes)
    
    def _generate_arterial_pressure(self, t):
        """Generar señal de presión arterial sintética"""
        # Presión base + variación cardíaca + ruido
        base_pressure = 90
        cardiac_variation = 30 * np.sin(2 * np.pi * 1.2 * t)  # ~72 bpm
        respiratory_variation = 5 * np.sin(2 * np.pi * 0.3 * t)  # ~18 rpm
        noise = np.random.normal(0, 2, len(t))
        
        return base_pressure + cardiac_variation + respiratory_variation + noise
    
    def _generate_ecg_signal(self, t):
        """Generar señal ECG sintética"""
        # ECG simplificado con componentes principales
        heart_rate = 1.2  # Hz (~72 bpm)
        
        # Componente principal (QRS)
        qrs_component = np.sin(2 * np.pi * heart_rate * t)
        
        # Componentes armónicos
        p_wave = 0.2 * np.sin(2 * np.pi * heart_rate * t - np.pi/4)
        t_wave = 0.3 * np.sin(2 * np.pi * heart_rate * t + np.pi/3)
        
        # Ruido de línea base
        noise = np.random.normal(0, 0.1, len(t))
        
        return qrs_component + p_wave + t_wave + noise
    
    def _generate_pleth_signal(self, t):
        """Generar señal de pletismografía sintética"""
        # Saturación de oxígeno base + variación
        base_spo2 = 97
        cardiac_variation = 2 * np.sin(2 * np.pi * 1.2 * t)
        drift = 1 * np.sin(2 * np.pi * 0.01 * t)  # Deriva lenta
        noise = np.random.normal(0, 0.5, len(t))
        
        return base_spo2 + cardiac_variation + drift + noise
    
    def create_mock_model_files(self):
        """Crear archivos de modelo mock para tests"""
        models_dir = os.path.join(self.base_dir, 'models')
        os.makedirs(models_dir, exist_ok=True)
        
        # Crear modelo sklearn mock
        from sklearn.ensemble import RandomForestClassifier
        import joblib
        
        # Modelo tabular simple
        tabular_model = RandomForestClassifier(n_estimators=10, random_state=42)
        # Entrenar con datos dummy
        X_dummy = np.random.randn(100, 3)
        y_dummy = np.random.randint(0, 2, 100)
        tabular_model.fit(X_dummy, y_dummy)
        
        tabular_path = os.path.join(models_dir, 'mock_tabular_model.joblib')
        joblib.dump(tabular_model, tabular_path)
        
        return models_dir
    
    def create_mock_config(self, models_dir):
        """Crear configuración mock de modelos"""
        import json
        
        config = [
            {
                "name": "Mock Predicción hTA",
                "path": os.path.join(models_dir, "mock_tabular_model.joblib"),
                "input_type": "tabular",
                "input_vars": ["Demo/ART", "Demo/ECG", "Demo/PLETH"],
                "output_var": "Prediccion_Mock",
                "window_size": 1
            }
        ]
        
        config_path = os.path.join(self.base_dir, 'model_mock.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_path
    
    def cleanup(self):
        """Limpiar archivos temporales"""
        if self.base_dir and os.path.exists(self.base_dir):
            shutil.rmtree(self.base_dir, ignore_errors=True)


def create_test_environment():
    """Crear entorno completo de prueba"""
    generator = MockDataGenerator()
    
    # Crear estructura completa
    records_dir = generator.create_mock_records_structure(num_days=2)
    models_dir = generator.create_mock_model_files()
    config_path = generator.create_mock_config(models_dir)
    
    return {
        'base_dir': generator.base_dir,
        'records_dir': records_dir,
        'models_dir': models_dir,
        'config_path': config_path,
        'generator': generator
    }


if __name__ == '__main__':
    # Test del generador de datos mock
    print("Creando entorno de prueba...")
    test_env = create_test_environment()
    
    print(f"Directorio base: {test_env['base_dir']}")
    print(f"Records: {test_env['records_dir']}")
    print(f"Models: {test_env['models_dir']}")
    print(f"Config: {test_env['config_path']}")
    
    # Listar archivos creados
    for root, dirs, files in os.walk(test_env['base_dir']):
        level = root.replace(test_env['base_dir'], '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files:
            print(f"{subindent}{file}")
    
    print("Limpiando archivos temporales...")
    test_env['generator'].cleanup()
    print("Archivos temporales eliminados.")
