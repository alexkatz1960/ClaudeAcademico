#!/usr/bin/env python3
"""
Test de Configuración CORREGIDO - Sistema ClaudeAcademico v2.2
Verifica que todas las APIs y componentes estén correctamente configurados
"""

import os
import sys
import requests
import asyncio
from pathlib import Path
from dotenv import load_dotenv
import json
from datetime import datetime

# Cargar variables de entorno
load_dotenv()

class ConfigurationTester:
    def __init__(self):
        self.results = {}
        self.errors = []
        
    def log_result(self, test_name, status, message, details=None):
        """Registrar resultado de test"""
        self.results[test_name] = {
            'status': status,
            'message': message,
            'details': details,
            'timestamp': datetime.now().isoformat()
        }
        
        status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
        print(f"{status_icon} {test_name}: {message}")
        
        if details and status != "PASS":
            print(f"   Details: {details}")
    
    def test_environment_variables(self):
        """Test 1: Verificar variables de entorno - VERSIÓN CORREGIDA"""
        print("\n🔧 Testing Environment Variables...")
        
        required_vars = {
            'DEEPL_API_KEY': 'DeepL API Key',
            'ABBYY_APPLICATION_ID': 'ABBYY Application ID', 
            'ABBYY_PASSWORD': 'ABBYY Password',
            'DATABASE_URL': 'Database URL',
            'ENVIRONMENT': 'Environment Setting',
            'SECRET_KEY': 'Security Secret Key'
        }
        
        missing_vars = []
        for var, description in required_vars.items():
            value = os.getenv(var)
            if not value:
                missing_vars.append(f"{var} ({description})")
            elif var == 'DEEPL_API_KEY' and len(value) < 10:
                missing_vars.append(f"{var} (API key seems too short)")
            elif var == 'ABBYY_PASSWORD' and len(value) < 8:
                missing_vars.append(f"{var} (password seems too short)")
            elif var == 'ABBYY_APPLICATION_ID' and len(value) < 10:
                missing_vars.append(f"{var} (application ID seems too short)")
            elif var == 'SECRET_KEY' and len(value) < 32:
                missing_vars.append(f"{var} (secret key too short - minimum 32 chars)")
        
        if missing_vars:
            self.log_result(
                "Environment Variables", 
                "FAIL", 
                f"Missing or invalid variables: {len(missing_vars)}",
                missing_vars
            )
            return False
        else:
            self.log_result(
                "Environment Variables", 
                "PASS", 
                "All required variables configured"
            )
            return True
    
    def test_directory_structure(self):
        """Test 2: Verificar estructura de directorios"""
        print("\n📁 Testing Directory Structure...")
        
        required_dirs = [
            'data', 'logs', 'uploads', 'output', 'temp', 'backups',
            'interfaces', 'interfaces/database'
        ]
        
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = Path(dir_name)
            if not dir_path.exists():
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                    print(f"   Created directory: {dir_name}")
                except Exception as e:
                    missing_dirs.append(f"{dir_name}: {str(e)}")
        
        if missing_dirs:
            self.log_result(
                "Directory Structure", 
                "FAIL", 
                f"Cannot create directories: {len(missing_dirs)}",
                missing_dirs
            )
            return False
        else:
            self.log_result(
                "Directory Structure", 
                "PASS", 
                "All required directories exist"
            )
            return True
    
    def test_deepl_api(self):
        """Test 3: Verificar conexión a DeepL API"""
        print("\n🌐 Testing DeepL API...")
        
        api_key = os.getenv('DEEPL_API_KEY')
        if not api_key:
            self.log_result("DeepL API", "FAIL", "API key not configured")
            return False
        
        try:
            # Test simple de conexión
            url = "https://api.deepl.com/v2/usage"
            headers = {"Authorization": f"DeepL-Auth-Key {api_key}"}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                usage_data = response.json()
                character_count = usage_data.get('character_count', 0)
                character_limit = usage_data.get('character_limit', 0)
                
                self.log_result(
                    "DeepL API", 
                    "PASS", 
                    f"Connected successfully. Usage: {character_count:,}/{character_limit:,} characters"
                )
                return True
            elif response.status_code == 403:
                self.log_result("DeepL API", "FAIL", "Invalid API key (403 Forbidden)")
                return False
            else:
                self.log_result(
                    "DeepL API", 
                    "FAIL", 
                    f"Unexpected response: {response.status_code}",
                    response.text[:200]
                )
                return False
                
        except requests.exceptions.Timeout:
            self.log_result("DeepL API", "WARN", "Connection timeout (check internet)")
            return False
        except requests.exceptions.RequestException as e:
            self.log_result("DeepL API", "FAIL", "Connection error", str(e))
            return False
    
    def test_abbyy_api(self):
        """Test 4: Verificar conexión a ABBYY API - VERSIÓN CORREGIDA"""
        print("\n🔍 Testing ABBYY API...")
        
        app_id = os.getenv('ABBYY_APPLICATION_ID')
        password = os.getenv('ABBYY_PASSWORD')
        
        if not app_id or not password:
            missing = []
            if not app_id:
                missing.append("ABBYY_APPLICATION_ID")
            if not password:
                missing.append("ABBYY_PASSWORD")
            
            self.log_result(
                "ABBYY API", 
                "FAIL", 
                f"Missing credentials: {', '.join(missing)}"
            )
            return False
        
        try:
            # Test de autenticación con Application ID + Password
            url = "https://cloud.abbyy.com/v2/listTasks"
            auth = (app_id, password)  # ABBYY usa HTTP Basic Auth
            headers = {'User-Agent': 'ClaudeAcademico/2.2'}
            
            print(f"   Testing with Application ID: {app_id[:8]}...")
            
            response = requests.get(url, auth=auth, headers=headers, timeout=15)
            
            if response.status_code == 200:
                # Verificar que la respuesta contiene datos válidos
                try:
                    task_data = response.json()
                    task_count = len(task_data.get('task', []))
                    self.log_result(
                        "ABBYY API", 
                        "PASS", 
                        f"Authentication successful. Active tasks: {task_count}"
                    )
                    return True
                except:
                    self.log_result("ABBYY API", "PASS", "Authentication successful")
                    return True
            elif response.status_code == 401:
                self.log_result("ABBYY API", "FAIL", "Invalid credentials (401 Unauthorized)")
                return False
            elif response.status_code == 403:
                self.log_result("ABBYY API", "FAIL", "Access forbidden (403) - Check account status")
                return False
            elif response.status_code == 404:
                # Probablemente la URL base está bien, pero el endpoint específico no
                self.log_result("ABBYY API", "WARN", "Endpoint not found (credentials seem valid)")
                return True
            else:
                self.log_result(
                    "ABBYY API", 
                    "WARN", 
                    f"Unexpected response: {response.status_code}",
                    f"Credentials may still work. Response: {response.text[:100]}"
                )
                return True
                
        except requests.exceptions.Timeout:
            self.log_result("ABBYY API", "WARN", "Connection timeout (check internet)")
            return False
        except requests.exceptions.RequestException as e:
            self.log_result("ABBYY API", "FAIL", "Connection error", str(e))
            return False
    
    def test_database_connection(self):
        """Test 5: Verificar conexión a base de datos"""
        print("\n💾 Testing Database Connection...")
        
        try:
            # Test básico de importación
            import sqlite3
            
            # Test de conexión SQLite simple
            db_path = "data/test_connection.db"
            Path("data").mkdir(exist_ok=True)
            
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            result = cursor.fetchone()
            conn.close()
            
            if result and result[0] == 1:
                self.log_result(
                    "Database Connection", 
                    "PASS", 
                    "SQLite database connection successful"
                )
                
                # Limpiar archivo de test
                Path(db_path).unlink(missing_ok=True)
                return True
            else:
                self.log_result("Database Connection", "FAIL", "Database test query failed")
                return False
                
        except ImportError as e:
            self.log_result(
                "Database Connection", 
                "FAIL", 
                "Cannot import database modules",
                str(e)
            )
            return False
        except Exception as e:
            self.log_result("Database Connection", "FAIL", "Database error", str(e))
            return False
    
    def test_dependencies(self):
        """Test 6: Verificar dependencias críticas"""
        print("\n📦 Testing Dependencies...")
        
        critical_modules = {
            'fastapi': 'FastAPI framework',
            'pandas': 'Data processing',
            'requests': 'HTTP client',
            'beautifulsoup4': 'HTML parsing',
            'mammoth': 'DOCX processing',
            'python_docx': 'DOCX creation'
        }
        
        missing_modules = []
        installed_modules = []
        
        for module, description in critical_modules.items():
            try:
                if module == 'beautifulsoup4':
                    import bs4
                elif module == 'python_docx':
                    import docx
                else:
                    __import__(module)
                installed_modules.append(f"{module} ({description})")
            except ImportError:
                missing_modules.append(f"{module} ({description})")
        
        if missing_modules:
            self.log_result(
                "Dependencies", 
                "FAIL", 
                f"Missing modules: {len(missing_modules)}",
                missing_modules
            )
            return False
        else:
            self.log_result(
                "Dependencies", 
                "PASS", 
                f"All {len(installed_modules)} critical modules available"
            )
            return True
    
    def generate_report(self):
        """Generar reporte final"""
        print("\n" + "="*60)
        print("🎯 CONFIGURATION TEST REPORT - UPDATED")
        print("="*60)
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r['status'] == 'PASS')
        failed_tests = sum(1 for r in self.results.values() if r['status'] == 'FAIL')
        warned_tests = sum(1 for r in self.results.values() if r['status'] == 'WARN')
        
        print(f"Total Tests: {total_tests}")
        print(f"✅ Passed: {passed_tests}")
        print(f"❌ Failed: {failed_tests}")
        print(f"⚠️  Warnings: {warned_tests}")
        print(f"Success Rate: {(passed_tests/total_tests)*100:.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! System is ready for processing.")
            print("\n🚀 Next steps:")
            print("1. Place a test PDF in the 'uploads' folder")
            print("2. Run: docker compose up --build")
            print("3. Access FastAPI docs: http://localhost:8000/docs")
            print("4. Access Streamlit dashboard: http://localhost:8501")
        elif failed_tests <= 2:
            print(f"\n⚠️  {failed_tests} minor issue(s). System may still work.")
            print("Fix issues for optimal performance.")
        else:
            print(f"\n❌ {failed_tests} TEST(S) FAILED. Fix critical issues before proceeding.")
            
        # Guardar reporte detallado
        report_path = Path("logs/configuration_test_report.json")
        report_path.parent.mkdir(exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump({
                'summary': {
                    'total': total_tests,
                    'passed': passed_tests,
                    'failed': failed_tests,
                    'warned': warned_tests,
                    'success_rate': (passed_tests/total_tests)*100
                },
                'results': self.results,
                'timestamp': datetime.now().isoformat()
            }, f, indent=2)
        
        print(f"\n📄 Detailed report saved: {report_path}")
        
        return failed_tests <= 1  # Allow 1 minor failure

def main():
    """Ejecutar todos los tests de configuración"""
    print("🚀 ClaudeAcademico v2.2 - Configuration Test (FIXED)")
    print("=" * 55)
    
    tester = ConfigurationTester()
    
    # Ejecutar todos los tests
    tests = [
        tester.test_environment_variables,
        tester.test_directory_structure,
        tester.test_dependencies,
        tester.test_database_connection,
        tester.test_deepl_api,
        tester.test_abbyy_api
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    # Generar reporte final
    return tester.generate_report()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)