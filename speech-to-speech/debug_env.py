#!/usr/bin/env python3
"""
Script específico para diagnosticar problemas con variables de entorno
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

def debug_env_file():
    """Diagnosticar problemas con el archivo .env"""
    print("🔍 DIAGNÓSTICO DETALLADO DEL ARCHIVO .env")
    print("="*60)
    
    # Verificar directorio actual
    current_dir = Path.cwd()
    print(f"📂 Directorio actual: {current_dir}")
    
    # Buscar archivos .env
    env_files = []
    for env_file in ['.env', '../.env', '../../.env']:
        env_path = Path(env_file)
        if env_path.exists():
            env_files.append(env_path.resolve())
            print(f"✅ Archivo .env encontrado: {env_path.resolve()}")
        else:
            print(f"❌ No encontrado: {env_path.resolve()}")
    
    print()
    
    if not env_files:
        print("❌ NO SE ENCONTRÓ NINGÚN ARCHIVO .env")
        print("\n📝 Crear archivo .env con el siguiente contenido:")
        print("GEMINI_API_KEY=tu_clave_aqui")
        print("ELEVENLABS_API_KEY=tu_clave_aqui")
        return
    
    # Analizar cada archivo .env encontrado
    for env_file in env_files:
        print(f"📄 ANALIZANDO: {env_file}")
        print("-" * 50)
        
        try:
            with open(env_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            print(f"📊 Total de líneas: {len(lines)}")
            
            # Analizar cada línea
            gemini_found = False
            elevenlabs_found = False
            
            for i, line in enumerate(lines, 1):
                line_clean = line.strip()
                if not line_clean or line_clean.startswith('#'):
                    continue
                
                if '=' in line_clean:
                    key, value = line_clean.split('=', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Verificar claves específicas
                    if key == 'GEMINI_API_KEY':
                        gemini_found = True
                        if value:
                            print(f"✅ Línea {i}: GEMINI_API_KEY = {'*' * min(10, len(value))}...")
                        else:
                            print(f"⚠️  Línea {i}: GEMINI_API_KEY está vacía")
                    
                    elif key == 'ELEVENLABS_API_KEY':
                        elevenlabs_found = True
                        if value:
                            print(f"✅ Línea {i}: ELEVENLABS_API_KEY = {'*' * min(10, len(value))}...")
                        else:
                            print(f"⚠️  Línea {i}: ELEVENLABS_API_KEY está vacía")
                    
                    else:
                        print(f"ℹ️  Línea {i}: {key} = {'*' * min(5, len(value))}...")
                else:
                    print(f"⚠️  Línea {i}: Formato incorrecto: {line_clean[:50]}...")
            
            print()
            print("📋 RESUMEN:")
            print(f"   GEMINI_API_KEY: {'✅ Encontrada' if gemini_found else '❌ NO encontrada'}")
            print(f"   ELEVENLABS_API_KEY: {'✅ Encontrada' if elevenlabs_found else '❌ NO encontrada'}")
            
        except Exception as e:
            print(f"❌ Error leyendo archivo: {e}")
        
        print()

def test_dotenv_loading():
    """Probar carga de variables con python-dotenv"""
    print("🔄 PROBANDO CARGA CON PYTHON-DOTENV")
    print("="*60)
    
    # Probar carga sin especificar archivo
    print("1. Carga automática...")
    result = load_dotenv()
    print(f"   Resultado: {result}")
    
    # Probar diferentes ubicaciones
    env_locations = ['.env', '../.env', '../../.env']
    
    for location in env_locations:
        env_path = Path(location)
        if env_path.exists():
            print(f"\n2. Cargando desde {env_path.resolve()}...")
            result = load_dotenv(env_path)
            print(f"   Resultado: {result}")
            break
    
    print()

def test_environment_variables():
    """Probar acceso a variables de entorno"""
    print("🌍 PROBANDO ACCESO A VARIABLES DE ENTORNO")
    print("="*60)
    
    # Variables que esperamos
    expected_vars = ['GEMINI_API_KEY', 'ELEVENLABS_API_KEY']
    
    print("Variables de entorno después de cargar .env:")
    for var in expected_vars:
        value = os.getenv(var)
        if value:
            print(f"✅ {var}: {'*' * min(10, len(value))}... (longitud: {len(value)})")
        else:
            print(f"❌ {var}: NO configurada")
    
    print()
    
    # Mostrar todas las variables que empiecen con las claves conocidas
    print("Todas las variables relacionadas:")
    all_env = dict(os.environ)
    related_vars = {k: v for k, v in all_env.items() 
                   if any(keyword in k.upper() for keyword in ['GEMINI', 'ELEVEN', 'GOOGLE', 'API'])}
    
    if related_vars:
        for var, value in related_vars.items():
            print(f"ℹ️  {var}: {'*' * min(5, len(value))}...")
    else:
        print("❌ No se encontraron variables relacionadas")
    
    print()

def suggest_fixes():
    """Sugerir soluciones basadas en el diagnóstico"""
    print("🔧 POSIBLES SOLUCIONES")
    print("="*60)
    
    print("Si GEMINI_API_KEY no se detecta:")
    print("1. Verificar que el archivo .env esté en el directorio correcto")
    print("2. Verificar que no haya espacios alrededor del '=' ")
    print("   ❌ Incorrecto: GEMINI_API_KEY = tu_clave")
    print("   ✅ Correcto:   GEMINI_API_KEY=tu_clave")
    print()
    print("3. Verificar que no haya caracteres especiales invisibles")
    print("4. Verificar que la clave API esté completa y válida")
    print()
    print("5. Formato correcto del archivo .env:")
    print("   GEMINI_API_KEY=AIzaSy...")
    print("   ELEVENLABS_API_KEY=sk_...")
    print()
    print("6. Si sigue fallando, intenta recrear el archivo .env desde cero")

def main():
    """Ejecutar diagnóstico completo"""
    print("🚀 DIAGNÓSTICO COMPLETO DE VARIABLES DE ENTORNO")
    print("="*70)
    print()
    
    debug_env_file()
    test_dotenv_loading()
    test_environment_variables()
    suggest_fixes()
    
    print("🏁 DIAGNÓSTICO COMPLETADO")
    print("="*70)

if __name__ == "__main__":
    main()