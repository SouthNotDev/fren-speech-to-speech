#!/usr/bin/env python3
"""
Speech to Speech System
Complete pipeline: Whisper STT -> Gemini LLM -> ElevenLabs TTS
"""

import os
import logging
import sounddevice as sd
import numpy as np
import torch
from time import perf_counter
import time
import sys
import webrtcvad
from transformers import pipeline
import google.generativeai as genai
from elevenlabs.client import ElevenLabs
from elevenlabs import stream
from collections import deque

# Import platform-specific modules
if os.name != 'nt':  # Unix/Linux only
    import termios
    import tty

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# STT imports
from faster_whisper import WhisperModel

# LLM imports - Using NEW Google Gen AI SDK for proper search grounding
from google import genai
from google.genai import types

# TTS imports
from elevenlabs.client import ElevenLabs

# Rich console for better output
try:
    from rich.console import Console
    console = Console()
except ImportError:
    console = None

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Language codes supported by ElevenLabs Turbo v2.5
SUPPORTED_LANGUAGE_CODES = {
    "en": "en", "es": "es", "fr": "fr", "de": "de", "it": "it",
    "pt": "pt", "pl": "pl", "tr": "tr", "ru": "ru", "nl": "nl",
    "cs": "cs", "ar": "ar", "zh": "zh", "ja": "ja", "hi": "hi", "ko": "ko"
}

# Note: Gemini 2.0 Flash has built-in Google Search capability
# No need to configure external search tools

class AudioDeviceSelector:
    """Interactive audio device selector with arrow key navigation"""
    
    def __init__(self):
        self.input_devices = []
        self.output_devices = []
        self.selected_input = None
        self.selected_output = None
        
    def get_audio_devices(self):
        """Get available audio devices"""
        devices = sd.query_devices()
        self.input_devices = []
        self.output_devices = []
        
        for i, device in enumerate(devices):
            device_info = {
                'index': i,
                'name': device['name'],
                'max_input_channels': device['max_input_channels'],
                'max_output_channels': device['max_output_channels'],
                'default_samplerate': device['default_samplerate']
            }
            
            if device['max_input_channels'] > 0:
                self.input_devices.append(device_info)
            if device['max_output_channels'] > 0:
                self.output_devices.append(device_info)
    
    def clear_screen(self):
        """Clear the console screen"""
        os.system('cls' if os.name == 'nt' else 'clear')
    
    def get_key(self):
        """Get a single keypress (works on Unix/Linux)"""
        if os.name == 'nt':  # Windows
            import msvcrt
            key = msvcrt.getch()
            if key == b'\xe0':  # Arrow keys on Windows
                key = msvcrt.getch()
                if key == b'H': return 'UP'
                elif key == b'P': return 'DOWN'
            elif key == b'\r': return 'ENTER'
            elif key == b'\x1b': return 'ESC'
            return key.decode('utf-8', errors='ignore')
        else:  # Unix/Linux
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(sys.stdin.fileno())
                key = sys.stdin.read(1)
                if key == '\x1b':  # Escape sequence
                    key += sys.stdin.read(2)
                    if key == '\x1b[A': return 'UP'
                    elif key == '\x1b[B': return 'DOWN'
                    elif key == '\x1b[C': return 'RIGHT'
                    elif key == '\x1b[D': return 'LEFT'
                elif key == '\r' or key == '\n': return 'ENTER'
                elif key == '\x1b': return 'ESC'
                return key
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def show_welcome_screen(self):
        """Show welcome screen and wait for Enter"""
        self.clear_screen()
        print("="*60)
        print("🎤 SISTEMA SPEECH-TO-SPEECH 🔊")
        print("="*60)
        print()
        print("🚀 ¡Bienvenido al sistema de conversación por voz!")
        print()
        print("   Características:")
        print("   • 🎯 Whisper STT (reconocimiento de voz)")
        print("   • 🧠 Gemini 2.0 LLM (inteligencia artificial)")
        print("   • 🔊 ElevenLabs TTS (síntesis de voz)")
        print("   • 🔍 Búsqueda automática en Google")
        print()
        print("📋 A continuación configuraremos los dispositivos de audio")
        print()
        print("🎯 Presiona ENTER para comenzar la configuración...")
        print("   (ESC para salir)")
        
        while True:
            key = self.get_key()
            if key == 'ENTER':
                break
            elif key == 'ESC':
                print("\n❌ Configuración cancelada")
                sys.exit(0)
    
    def show_device_menu(self, devices, device_type, selected_index=0):
        """Show device selection menu"""
        self.clear_screen()
        print("="*60)
        print(f"🔧 SELECCIÓN DE {device_type.upper()}")
        print("="*60)
        print()
        print(f"📱 Selecciona tu {device_type}:")
        print(f"   (Usa ↑↓ para navegar, ENTER para seleccionar)")
        print()
        
        for i, device in enumerate(devices):
            marker = "→ " if i == selected_index else "  "
            highlight = "🔹" if i == selected_index else "🔸"
            print(f"{marker}{highlight} {device['name']}")
            if i == selected_index:
                print(f"     └─ Canales: {device['max_input_channels'] if device_type == 'micrófono' else device['max_output_channels']}")
                print(f"     └─ Frecuencia: {int(device['default_samplerate'])} Hz")
        
        print()
        print("🎯 ENTER: Seleccionar | ↑↓: Navegar | ESC: Salir")
    
    def select_device(self, devices, device_type):
        """Interactive device selection with arrow keys"""
        if not devices:
            print(f"❌ No se encontraron dispositivos de {device_type}")
            return None
        
        selected_index = 0
        
        while True:
            self.show_device_menu(devices, device_type, selected_index)
            
            key = self.get_key()
            
            if key == 'UP':
                selected_index = (selected_index - 1) % len(devices)
            elif key == 'DOWN':
                selected_index = (selected_index + 1) % len(devices)
            elif key == 'ENTER':
                selected_device = devices[selected_index]
                self.clear_screen()
                print("="*60)
                print("✅ DISPOSITIVO SELECCIONADO")
                print("="*60)
                print()
                print(f"🎯 {device_type.capitalize()}: {selected_device['name']}")
                print()
                print("⏳ Continuando en 2 segundos...")
                time.sleep(2)
                return selected_device
            elif key == 'ESC':
                print("\n❌ Selección cancelada")
                sys.exit(0)
    
    def run_device_selection(self):
        """Run the complete device selection process"""
        # Welcome screen
        self.show_welcome_screen()
        
        # Get available devices
        print("\n🔍 Detectando dispositivos de audio...")
        self.get_audio_devices()
        
        # Select output device (speakers)
        print("\n🔊 Configurando parlantes...")
        time.sleep(1)
        self.selected_output = self.select_device(self.output_devices, "parlante")
        
        # Select input device (microphone)  
        print("\n🎤 Configurando micrófono...")
        time.sleep(1)
        self.selected_input = self.select_device(self.input_devices, "micrófono")
        
        # Show final configuration
        self.clear_screen()
        print("="*60)
        print("🎉 CONFIGURACIÓN COMPLETADA")
        print("="*60)
        print()
        print("📋 Dispositivos seleccionados:")
        print(f"   🔊 Parlante: {self.selected_output['name']}")
        print(f"   🎤 Micrófono: {self.selected_input['name']}")
        print()
        print("🚀 Iniciando sistema speech-to-speech...")
        print()
        time.sleep(3)
        
        return self.selected_input['index'], self.selected_output['index']

class SpeechToSpeechSystem:
    """Complete Speech-to-Speech system in a single class"""
    
    def __init__(self, input_device=None, output_device=None):
        # Audio device configuration
        self.input_device = input_device
        self.output_device = output_device
        
        # Configuration
        self.config = {
            'stt_model': 'small',
            'stt_language': 'es',
            'stt_device': 'auto',
            'stt_compute_type': 'int8',
            'llm_model': 'gemini-2.0-flash-exp',
            'llm_language': 'es',
            'llm_temperature': 0.7,
            'llm_max_tokens': 200,
            'llm_enable_search': True,
            'tts_voice_id': 'b2htR0pMe28pYwCY9gnP',
            'tts_model_id': 'eleven_turbo_v2_5',
            'tts_voice_settings': {
                "stability": 0.7,
                "similarity_boost": 0.8,
                "style": 0.0,
                "use_speaker_boost": True
            },
            'sample_rate': 16000,
            'channels': 1,
            'tts_sample_rate': 22050,
            'vad_threshold': 0.05,  # Menos sensible para evitar cortes prematuros
            'min_silence_ms': 2000,  # 2 segundos de silencio antes de cortar (permite pausas naturales)
            'max_wait_seconds': 45,  # Más tiempo para preguntas largas
            'log_level': 'INFO',
            'continuous_mode': True
        }
        
        # Update logging level
        log_level = getattr(logging, self.config['log_level'].upper(), logging.INFO)
        logging.getLogger().setLevel(log_level)
        
        # Initialize components
        self.stt_model = None
        self.llm_model = None
        self.tts_client = None
        self.vad_model = None
        
        # Chat history
        self.chat_history = []
        self.max_chat_history = 10
        
        # System prompt for Gemini
        self.system_prompt = """
Eres un asistente de conversación inteligente en español diseñado para mantener diálogos naturales y fluidos. 

CONTEXTO:
- Este es un sistema de conversación por voz en tiempo real
- El usuario habla contigo a través de micrófono y escucha tus respuestas
- Tu objetivo es simular una conversación humana natural

COMPORTAMIENTO:
- Responde de manera conversacional y natural
- Mantén respuestas concisas pero informativas (máximo 2-3 oraciones)
- Usa un tono amigable y cercano
- Adapta tu registro al contexto de la conversación

BÚSQUEDA EN TIEMPO REAL:
- SIEMPRE usa Google Search para consultas sobre:
  * Fecha y hora actual ("¿qué día es hoy?", "¿qué hora es?")
  * Noticias recientes ("¿qué noticias hay?", "noticias de hoy")
  * Información actualizada (clima, deportes, eventos actuales)
  * Precios, cotizaciones, stocks
  * Cualquier dato que pueda cambiar con el tiempo

- Para preguntas generales de conocimiento estático puedes usar tu conocimiento base

FORMATO DE RESPUESTA:
- Responde en español natural y conversacional
- No uses markdown o formato especial
- Habla como en una conversación presencial
- Evita respuestas extremadamente largas (más de 30 segundos de audio)

FECHA ACTUAL:
- Nunca asumas fechas. SIEMPRE busca la fecha actual cuando te pregunten.
"""
        
        # Initialize system
        self.initialize_system()
    
    def initialize_system(self):
        """Initialize all components"""
        logger.info("Initializing Speech-to-Speech system...")
        
        # Check environment variables
        self.check_environment()
        
        # Initialize components
        self.initialize_vad()
        self.initialize_stt()
        self.initialize_llm()
        self.initialize_tts()
        
        logger.info("Speech-to-Speech system ready!")
    
    def check_environment(self):
        """Check required environment variables"""
        required_vars = ['GEMINI_API_KEY', 'ELEVENLABS_API_KEY']
        missing_vars = [var for var in required_vars if not os.getenv(var)]
        
        if missing_vars:
            logger.error(f"Missing environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing environment variables: {missing_vars}")
        
        logger.info("Environment variables verified")
    
    def initialize_vad(self):
        """Initialize Voice Activity Detection"""
        logger.info("Loading VAD model...")
        self.vad_model, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
        logger.info("VAD model loaded")
    
    def initialize_stt(self):
        """Initialize Speech-to-Text"""
        logger.info(f"Loading STT model: {self.config['stt_model']}")
        os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
        self.stt_model = WhisperModel(
            self.config['stt_model'],
            device=self.config['stt_device'],
            compute_type=self.config['stt_compute_type']
        )
        logger.info("STT model loaded")
    
    def initialize_llm(self):
        """Initialize Language Model with Google Search grounding"""
        logger.info(f"Initializing LLM: {self.config['llm_model']}")
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("❌ GEMINI_API_KEY not found in environment variables")
            print("❌ Error: GEMINI_API_KEY no configurada")
            print("💡 Verificar archivo .env con formato: GEMINI_API_KEY=AIzaSy...")
            raise ValueError("GEMINI_API_KEY not configured")
        
        # Configure NEW SDK client
        self.client = genai.Client(api_key=api_key)
        
        # Configure Google Search tool for grounding
        if self.config['llm_enable_search']:
            self.google_search_tool = types.Tool(
                google_search=types.GoogleSearch()
            )
            
            # Model configuration with search tools
            self.generation_config = types.GenerateContentConfig(
                tools=[self.google_search_tool],
                response_modalities=["TEXT"],
                temperature=self.config['llm_temperature'],
                max_output_tokens=self.config['llm_max_tokens']
            )
            logger.info("✅ Google Search grounding configured")
        else:
            # Model configuration without search
            self.generation_config = types.GenerateContentConfig(
                response_modalities=["TEXT"],
                temperature=self.config['llm_temperature'],
                max_output_tokens=self.config['llm_max_tokens']
            )
            logger.info("Google Search disabled in config")
        
        # Test the model
        try:
            test_response = self.client.models.generate_content(
                model=self.config['llm_model'],
                contents="Hola",
                config=self.generation_config
            )
            logger.info("✅ LLM initialized and tested successfully")
        except Exception as e:
            logger.error(f"❌ LLM test failed: {e}")
            raise
    
    def initialize_tts(self):
        """Initialize Text-to-Speech"""
        logger.info("Initializing TTS...")
        api_key = os.getenv("ELEVENLABS_API_KEY")
        self.tts_client = ElevenLabs(api_key=api_key)
        
        # Test TTS
        warmup_params = {
            "text": "Hola",
            "voice_id": self.config['tts_voice_id'],
            "model_id": self.config['tts_model_id'],
            "output_format": "pcm_22050",
            "voice_settings": self.config['tts_voice_settings']
        }
        
        if self.config['tts_model_id'] in ["eleven_turbo_v2_5", "eleven_flash_v2_5"]:
            warmup_params["language_code"] = SUPPORTED_LANGUAGE_CODES[self.config['llm_language']]
        
        audio_stream = self.tts_client.text_to_speech.stream(**warmup_params)
        for _ in audio_stream:
            pass
        
        logger.info("TTS initialized and tested")
    
    def detect_speech_with_vad(self):
        """Detect speech using VAD with continuous buffer to capture complete speech (SOLUTION FOR MISSING SYLLABLES)"""
        logger.info("Listening for speech...")
        
        from collections import deque
        
        # Continuous buffer to capture speech that starts before VAD detection
        buffer_duration = 2.0  # Keep 2 seconds of pre-speech audio
        buffer_size = int(buffer_duration * self.config['sample_rate'] / 512)
        continuous_buffer = deque(maxlen=buffer_size)
        
        speech_chunks = []
        speech_detected = False
        silence_chunks = 0
        required_silence_chunks = int(self.config['min_silence_ms'] / (512 / self.config['sample_rate'] * 1000))
        
        start_time = time.time()
        
        # Try continuous audio stream first (better method)
        try:
            def audio_callback(indata, frames, time, status):
                if status:
                    logger.warning(f"Audio callback status: {status}")
                chunk = indata.flatten().copy()
                continuous_buffer.append(chunk)
            
            with sd.InputStream(
                callback=audio_callback,
                device=self.input_device,
                channels=1,
                samplerate=self.config['sample_rate'],
                blocksize=512,
                dtype='float32'
            ):
                logger.info("🎤 Continuous listening active (captures ALL syllables)")
                
                while time.time() - start_time < self.config['max_wait_seconds']:
                    if len(continuous_buffer) > 0:
                        # Get latest chunk for VAD processing
                        latest_chunk = continuous_buffer[-1]
                        chunk_tensor = torch.from_numpy(latest_chunk)
                        
                        with torch.no_grad():
                            speech_prob = self.vad_model(chunk_tensor, self.config['sample_rate']).item()
                        
                        if speech_prob > self.config['vad_threshold']:
                            if not speech_detected:
                                # Speech just started - include entire buffer for complete capture
                                logger.info(f"Speech started! (prob: {speech_prob:.3f})")
                                speech_chunks = list(continuous_buffer)  # Include pre-speech context
                                speech_detected = True
                            else:
                                # Continue capturing speech
                                speech_chunks.append(latest_chunk)
                            
                            silence_chunks = 0
                            
                        else:
                            if speech_detected:
                                # Add silence chunk (natural speech includes pauses)
                                speech_chunks.append(latest_chunk)
                                silence_chunks += 1
                                
                                # Check if speech has ended
                                if silence_chunks >= required_silence_chunks:
                                    if len(speech_chunks) > 0:
                                        full_audio = np.concatenate(speech_chunks)
                                        duration_ms = len(full_audio) / self.config['sample_rate'] * 1000
                                        logger.info(f"Speech ended: {duration_ms:.0f}ms")
                                        return full_audio
                                    break
                    
                    time.sleep(0.01)  # Small delay to prevent CPU overload
                    
        except Exception as e:
            logger.warning(f"Continuous stream failed: {e}, falling back to chunk recording")
            # Fallback to original method if continuous stream fails
            return self._fallback_chunk_recording(start_time)
        
        # Return partial speech if timeout occurs
        if speech_chunks:
            logger.info("Timeout reached, returning captured speech")
            return np.concatenate(speech_chunks)
        
        return None
    
    def _fallback_chunk_recording(self, start_time):
        """Fallback to chunk-based recording if continuous stream fails"""
        audio_buffer = []
        speech_detected = False
        silence_chunks = 0
        required_silence_chunks = int(self.config['min_silence_ms'] / (512 / self.config['sample_rate'] * 1000))
        
        logger.warning("⚠️  Using fallback recording (may miss first syllables)")
        
        while time.time() - start_time < self.config['max_wait_seconds']:
            # Record 512 samples (required by VAD)
            chunk = sd.rec(512, samplerate=self.config['sample_rate'], channels=1, dtype='float32', device=self.input_device)
            sd.wait()
            chunk = chunk.flatten()
            
            # Process with VAD
            chunk_tensor = torch.from_numpy(chunk)
            with torch.no_grad():
                speech_prob = self.vad_model(chunk_tensor, self.config['sample_rate']).item()
            
            if speech_prob > self.config['vad_threshold']:
                if not speech_detected:
                    logger.info(f"Speech started! (prob: {speech_prob:.3f})")
                    speech_detected = True
                
                audio_buffer.append(chunk)
                silence_chunks = 0
            else:
                if speech_detected:
                    audio_buffer.append(chunk)
                    silence_chunks += 1
                    
                    if silence_chunks >= required_silence_chunks:
                        if len(audio_buffer) > 0:
                            full_audio = np.concatenate(audio_buffer)
                            duration_ms = len(full_audio) / self.config['sample_rate'] * 1000
                            logger.info(f"Speech ended: {duration_ms:.0f}ms")
                            return full_audio
                        break
        
        return None
    
    def process_stt(self, audio_data):
        """Process speech to text"""
        if audio_data is None:
            return None
            
        logger.info("Processing STT...")
        start_time = perf_counter()
        
        try:
            gen_kwargs = {
                "language": self.config['stt_language'],
                "without_timestamps": True
            }
            
            segments, info = self.stt_model.transcribe(audio_data, **gen_kwargs)
            output_text = [segment.text for segment in segments]
            text = " ".join(output_text).strip()
            
            processing_time = perf_counter() - start_time
            
            if text:
                if console:
                    console.print(f"[yellow]USER: {text}")
                else:
                    print(f"USER: {text}")
                logger.info(f"STT ({processing_time:.2f}s): '{text}'")
                return text
            else:
                return None
                
        except Exception as e:
            logger.error(f"Error in STT: {e}")
            return None
    
    def process_llm(self, text):
        """Process text through LLM with Google Search grounding"""
        if not text:
            return None
            
        logger.info("Processing LLM...")
        start_time = perf_counter()
        
        try:
            # Add to chat history
            self.add_to_chat_history("user", text)
            
            # Create the user prompt with system instructions included
            # Note: New SDK doesn't support separate system role, so we include it in user message
            full_prompt = f"""
{self.system_prompt}

Contexto de conversación previa:
{' '.join([f"{msg['role']}: {msg['content']}" for msg in self.chat_history[-3:]])}

Usuario: {text}
"""
            
            # Generate response using NEW SDK with search grounding
            response = self.client.models.generate_content(
                model=self.config['llm_model'],
                contents=[{"role": "user", "parts": [{"text": full_prompt}]}],
                config=self.generation_config
            )
            
            generated_text = response.text if response.text else ""
            processing_time = perf_counter() - start_time
            
            # Check if grounding was used
            if hasattr(response, 'candidates') and response.candidates:
                candidate = response.candidates[0]
                if hasattr(candidate, 'grounding_metadata') and candidate.grounding_metadata:
                    logger.info("✅ Google Search was used for grounding")
                    if hasattr(candidate.grounding_metadata, 'web_search_queries'):
                        queries = candidate.grounding_metadata.web_search_queries
                        logger.info(f"Search queries: {queries}")
            
            if generated_text.strip():
                # Add to chat history
                self.add_to_chat_history("assistant", generated_text)
                
                if console:
                    console.print(f"[blue]ASSISTANT: {generated_text}")
                else:
                    print(f"ASSISTANT: {generated_text}")
                logger.info(f"LLM ({processing_time:.2f}s)")
                return generated_text
            else:
                logger.warning("Empty response from LLM")
                return "Lo siento, no recibí una respuesta válida."
                
        except Exception as e:
            logger.error(f"Error in LLM: {e}")
            return "Lo siento, hubo un error al procesar tu mensaje."
    
    def process_tts(self, text):
        """Process text to speech"""
        if not text:
            return []
            
        logger.info("Processing TTS...")
        start_time = perf_counter()
        
        try:
            # Prepare streaming parameters
            stream_params = {
                "text": text,
                "voice_id": self.config['tts_voice_id'],
                "model_id": self.config['tts_model_id'],
                "output_format": "pcm_22050",
                "voice_settings": self.config['tts_voice_settings']
            }
            
            # Add language code if supported
            if (self.config['tts_model_id'] in ["eleven_turbo_v2_5", "eleven_flash_v2_5"] and 
                self.config['llm_language'] in SUPPORTED_LANGUAGE_CODES):
                stream_params["language_code"] = SUPPORTED_LANGUAGE_CODES[self.config['llm_language']]
            
            # Stream audio
            audio_stream = self.tts_client.text_to_speech.stream(**stream_params)
            
            audio_chunks = []
            buffer = b''
            blocksize = 512
            sample_width = 2  # int16
            
            for chunk in audio_stream:
                if chunk:
                    buffer += chunk
                    
                    while len(buffer) >= blocksize * sample_width:
                        audio_block = buffer[:blocksize * sample_width]
                        buffer = buffer[blocksize * sample_width:]
                        
                        audio_array = np.frombuffer(audio_block, dtype=np.int16)
                        audio_chunks.append(audio_array)
            
            # Handle remaining buffer
            if len(buffer) >= sample_width:
                audio_array = np.frombuffer(buffer, dtype=np.int16)
                audio_chunks.append(audio_array)
            
            processing_time = perf_counter() - start_time
            logger.info(f"TTS ({processing_time:.2f}s): {len(audio_chunks)} chunks")
            
            return audio_chunks
            
        except Exception as e:
            logger.error(f"Error in TTS: {e}")
            return []
    
    def play_audio(self, audio_chunks):
        """Play audio chunks"""
        if not audio_chunks:
            return
            
        try:
            # Concatenate all chunks
            full_audio = np.concatenate(audio_chunks)
            
            # Convert to float32
            if full_audio.dtype != np.float32:
                full_audio = full_audio.astype(np.float32) / 32768.0
            
            # Play audio with device compatibility
            logger.info(f"Playing audio: {len(full_audio)} samples")
            
            try:
                # Try original sample rate first
                sd.play(full_audio, samplerate=self.config['tts_sample_rate'], device=self.output_device)
                sd.wait()
                logger.info("Audio playback complete")
                
            except Exception as e:
                logger.warning(f"Playback failed with {self.config['tts_sample_rate']} Hz: {e}")
                
                # Try device default sample rate as fallback
                try:
                    device_info = sd.query_devices(self.output_device)
                    device_rate = int(device_info['default_samplerate'])
                    logger.info(f"Retrying with device rate: {device_rate} Hz")
                    
                    # Simple resampling if needed
                    if device_rate != self.config['tts_sample_rate']:
                        ratio = device_rate / self.config['tts_sample_rate']
                        new_length = int(len(full_audio) * ratio)
                        resampled = np.interp(
                            np.linspace(0, len(full_audio)-1, new_length),
                            np.arange(len(full_audio)),
                            full_audio
                        )
                        full_audio = resampled.astype(np.float32)
                    
                    sd.play(full_audio, samplerate=device_rate, device=self.output_device)
                    sd.wait()
                    logger.info("Audio playback complete (resampled)")
                    
                except Exception as retry_error:
                    logger.error(f"Audio playback failed completely: {retry_error}")
                    # Try with system default device as last resort
                    try:
                        sd.play(full_audio, samplerate=48000)  # Most common default
                        sd.wait()
                        logger.info("Audio playback complete (system default)")
                    except:
                        logger.error("All audio playback attempts failed")
            
        except Exception as e:
            logger.error(f"Error playing audio: {e}")
    
    def add_to_chat_history(self, role, content):
        """Add message to chat history"""
        self.chat_history.append({"role": role, "content": content})
        if len(self.chat_history) > self.max_chat_history:
            self.chat_history.pop(0)
    
    def process_speech_to_speech(self, audio_data):
        """Complete speech-to-speech pipeline"""
        if audio_data is None:
            return
            
        total_start = perf_counter()
        
        try:
            # STT
            text = self.process_stt(audio_data)
            if not text:
                return
            
            # LLM
            response = self.process_llm(text)
            if not response:
                return
            
            # TTS
            audio_chunks = self.process_tts(response)
            if not audio_chunks:
                return
            
            # Play audio
            self.play_audio(audio_chunks)
            
            total_time = perf_counter() - total_start
            logger.info(f"Total pipeline: {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"Error in speech-to-speech pipeline: {e}")
    
    def run_continuous(self):
        """Run in continuous mode"""
        logger.info("Starting continuous speech-to-speech mode")
        logger.info("Speak naturally - system will detect your voice!")
        
        try:
            while True:
                # Detect speech
                audio_data = self.detect_speech_with_vad()
                
                # Process if speech detected
                if audio_data is not None:
                    self.process_speech_to_speech(audio_data)
                else:
                    # Brief pause before listening again
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            logger.info("Stopping continuous mode...")
    
    def run_manual(self):
        """Run in manual mode"""
        logger.info("Manual speech-to-speech mode")
        logger.info("Press Enter to start recording")
        
        try:
            while True:
                input("Press Enter to record (Ctrl+C to quit): ")
                
                # Record for fixed duration
                duration = 5
                logger.info(f"Recording for {duration} seconds...")
                
                audio_data = sd.rec(
                    int(duration * self.config['sample_rate']),
                    samplerate=self.config['sample_rate'],
                    channels=1,
                    dtype='float32',
                    device=self.input_device
                ).flatten()
                
                sd.wait()
                logger.info("Recording complete")
                
                # Process the recording
                self.process_speech_to_speech(audio_data)
                
        except KeyboardInterrupt:
            logger.info("Stopping manual mode...")
    
    def run(self):
        """Run the system"""
        if self.config['continuous_mode']:
            self.run_continuous()
        else:
            self.run_manual()

def main():
    """Main entry point"""
    try:
        selector = AudioDeviceSelector()
        input_device, output_device = selector.run_device_selection()
        system = SpeechToSpeechSystem(input_device, output_device)
        system.run()
    except Exception as e:
        logger.error(f"Failed to start system: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()