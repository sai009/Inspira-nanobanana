#!/usr/bin/env python3
"""
Comprehensive Character Studio & Storyboard Generator
Combines the best features from v6.py and c4.py with enhanced architecture.
"""

import os
import json
import base64
import uuid
import shutil
import zipfile
import tempfile
import requests
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, asdict

import gradio as gr
from PIL import Image
from io import BytesIO
import google.generativeai as genai

# ElevenLabs Integration
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# =============================================================================
# CONFIGURATION & SETUP
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
PROJECTS_DIR = DATA_DIR / "projects"
CHARACTERS_DIR = DATA_DIR / "characters"
SCENES_DIR = DATA_DIR / "scenes"
STORYBOARDS_DIR = DATA_DIR / "storyboards"
EXPORTS_DIR = DATA_DIR / "exports"

# Audio directories for ElevenLabs integration
AUDIO_DIR = BASE_DIR / "audio_output"
VOICE_DIR = AUDIO_DIR / "voices"
DIALOGUE_DIR = AUDIO_DIR / "dialogue"
MUSIC_DIR = AUDIO_DIR / "music"
SFX_DIR = AUDIO_DIR / "sound_effects"

# Create directories
for directory in [DATA_DIR, PROJECTS_DIR, CHARACTERS_DIR, SCENES_DIR, STORYBOARDS_DIR, EXPORTS_DIR,
                 AUDIO_DIR, VOICE_DIR, DIALOGUE_DIR, MUSIC_DIR, SFX_DIR]:
    directory.mkdir(exist_ok=True, parents=True)

# ElevenLabs API Configuration
ELEVENLABS_API_BASE = "https://api.elevenlabs.io/v1"
DEFAULT_VOICE_SETTINGS = {
    "stability": 0.5,
    "similarity_boost": 0.75,
    "style": 0.0,
    "use_speaker_boost": True
}

def get_project_character_dir(project_id: str, char_id: str) -> Path:
    """Get character directory within a specific project"""
    project_char_dir = DATA_DIR / "projects" / project_id / "characters" / char_id
    project_char_dir.mkdir(exist_ok=True, parents=True)
    (project_char_dir / "references").mkdir(exist_ok=True)
    return project_char_dir

# Models - try multiple image model names as fallback
POSSIBLE_IMAGE_MODELS = [
    "gemini-2.5-flash-image-preview",
    "gemini-2.0-flash-image-preview", 
    "gemini-1.5-flash-image-preview",
    "imagen-3.0-generate-001",
    "imagegeneration@002"
]
IMAGE_MODEL = POSSIBLE_IMAGE_MODELS[0]  # Default to first
TEXT_MODEL = "gemini-2.5-flash"

# =============================================================================
# DATA MODELS
# =============================================================================

@dataclass
class Character:
    """Enhanced Character model with full feature support"""
    id: str
    name: str
    description: str
    personality: str = ""
    background: str = ""
    style_notes: str = ""
    tags: List[str] = None
    references: List[str] = None
    chat_history: List[Tuple[str, str]] = None  # (role, message) pairs
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.references is None:
            self.references = []
        if self.chat_history is None:
            self.chat_history = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()
    
    @property
    def dir(self) -> Path:
        """Character directory path"""
        char_dir = CHARACTERS_DIR / self.id
        char_dir.mkdir(exist_ok=True)
        (char_dir / "references").mkdir(exist_ok=True)
        return char_dir
    
    @property
    def references_dir(self) -> Path:
        """References directory for this character"""
        return self.dir / "references"
    
    def get_reference_paths(self) -> List[Path]:
        """Get full paths for reference images"""
        return [self.references_dir / ref for ref in self.references if (self.references_dir / ref).exists()]
    
    def get_prompt_header(self) -> str:
        """Generate consistent character prompt header for image generation"""
        header = f"""You are generating a consistent character for visual storytelling.
Character name: {self.name or '[unnamed]'}
Description: {self.description or '[none provided]'}
Personality: {self.personality or '[none provided]'}
Background: {self.background or '[none provided]'}
Visual style: {self.style_notes or '[none provided]'}

Keep facial features, hair, clothing style, and body proportions consistent across all images.
Pay attention to any reference images provided to maintain visual consistency."""
        return header

@dataclass
class Scene:
    """Scene/panel data model"""
    id: str
    instruction: str
    characters: List[str] = None
    image_path: str = ""
    created_at: str = ""
    
    def __post_init__(self):
        if self.characters is None:
            self.characters = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

@dataclass
class Project:
    """Enhanced Project model"""
    name: str
    id: str
    description: str = ""
    character_ids: List[str] = None
    scenes: List[Scene] = None
    storyboards: List[str] = None
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if self.character_ids is None:
            self.character_ids = []
        if self.scenes is None:
            self.scenes = []
        if self.storyboards is None:
            self.storyboards = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        self.updated_at = datetime.now().isoformat()

@dataclass
class AppConfig:
    """Application configuration"""
    api_key: str = ""
    elevenlabs_api_key: str = ""
    last_project: str = ""
    auto_save: bool = True
    image_quality: str = "high"
    export_format: str = "png"

@dataclass
class VoiceProfile:
    """Voice profile for characters"""
    id: str
    character_name: str
    voice_id: str  # ElevenLabs voice ID
    voice_name: str
    settings: Dict[str, Any]
    created_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

@dataclass
class AudioScene:
    """Audio scene with dialogue, music, and sound effects"""
    id: str
    scene_description: str
    characters: List[str]
    dialogue_lines: List[Dict[str, str]]  # {"character": "name", "text": "dialogue", "audio_file": "path"}
    music_file: str = ""
    sound_effects: List[Dict[str, str]] = None  # {"description": "...", "audio_file": "path"}
    created_at: str = ""
    
    def __post_init__(self):
        if self.sound_effects is None:
            self.sound_effects = []
        if not self.created_at:
            self.created_at = datetime.now().isoformat()

# =============================================================================
# DATA PERSISTENCE LAYER
# =============================================================================

class DataManager:
    """Handles all data persistence operations"""
    
    def __init__(self):
        self.config_path = DATA_DIR / "config.json"
        self.config = self.load_config()
    
    def load_config(self) -> AppConfig:
        """Load application configuration"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    data = json.load(f)
                return AppConfig(**data)
            except Exception:
                pass
        return AppConfig()
    
    def save_config(self, config: AppConfig):
        """Save application configuration"""
        self.config = config
        with open(self.config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2)
    
    def save_character(self, character: Character) -> Path:
        """Save character to file"""
        character.updated_at = datetime.now().isoformat()
        char_file = character.dir / "character.json"
        with open(char_file, 'w') as f:
            json.dump(asdict(character), f, indent=2)
        return char_file
    
    def load_character(self, char_id: str) -> Optional[Character]:
        """Load character from file"""
        char_file = CHARACTERS_DIR / char_id / "character.json"
        if char_file.exists():
            try:
                with open(char_file, 'r') as f:
                    data = json.load(f)
                return Character(**data)
            except Exception as e:
                print(f"Error loading character {char_id}: {e}")
        return None
    
    def list_characters(self) -> List[Character]:
        """List all available characters"""
        characters = []
        for char_dir in CHARACTERS_DIR.iterdir():
            if char_dir.is_dir():
                char = self.load_character(char_dir.name)
                if char:
                    characters.append(char)
        return sorted(characters, key=lambda c: c.name.lower())
    
    def delete_character(self, char_id: str) -> bool:
        """Delete character and all associated data"""
        char_dir = CHARACTERS_DIR / char_id
        if char_dir.exists():
            try:
                shutil.rmtree(char_dir)
                return True
            except Exception as e:
                print(f"Error deleting character: {e}")
        return False
    
    def save_project(self, project: Project) -> Path:
        """Save project to file"""
        project.updated_at = datetime.now().isoformat()
        # Convert Scene objects to dicts for JSON serialization
        project_dict = asdict(project)
        project_file = PROJECTS_DIR / f"{project.id}.json"
        with open(project_file, 'w') as f:
            json.dump(project_dict, f, indent=2)
        return project_file
    
    def load_project(self, project_id: str) -> Optional[Project]:
        """Load project from file"""
        project_file = PROJECTS_DIR / f"{project_id}.json"
        if project_file.exists():
            try:
                with open(project_file, 'r') as f:
                    data = json.load(f)
                # Convert scene dicts back to Scene objects
                if 'scenes' in data:
                    data['scenes'] = [Scene(**scene) for scene in data['scenes']]
                return Project(**data)
            except Exception as e:
                print(f"Error loading project {project_id}: {e}")
        return None
    
    def list_projects(self) -> List[Project]:
        """List all available projects"""
        projects = []
        for project_file in PROJECTS_DIR.glob("*.json"):
            project = self.load_project(project_file.stem)
            if project:
                projects.append(project)
        return sorted(projects, key=lambda p: p.updated_at, reverse=True)
    
    def delete_project(self, project_id: str) -> bool:
        """Delete project file"""
        project_file = PROJECTS_DIR / f"{project_id}.json"
        if project_file.exists():
            try:
                project_file.unlink()
                return True
            except Exception as e:
                print(f"Error deleting project: {e}")
        return False

# =============================================================================
# IMAGE GENERATION ENGINE
# =============================================================================

class ImageGenerator:
    """Enhanced image generation with Gemini"""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        if api_key:
            genai.configure(api_key=api_key)
    
    def configure_api(self, api_key: str) -> Tuple[bool, str]:
        """Configure Gemini API key"""
        try:
            genai.configure(api_key=api_key)
            self.api_key = api_key
            
            # Test the API with text model first
            text_model = genai.GenerativeModel(TEXT_MODEL)
            test_response = text_model.generate_content("Hello, this is a test.")
            
            # Test image model access
            try:
                image_model = genai.GenerativeModel(IMAGE_MODEL)
                return True, "✅ API Key validated! Both text and image generation models are accessible."
            except Exception as img_error:
                return True, f"✅ API Key validated for text. ⚠️ Image model issue: {str(img_error)}"
                
        except Exception as e:
            return False, f"❌ API validation failed: {str(e)}"
    
    def generate_image(self, prompt: str, reference_images: List[Image.Image] = None) -> Tuple[Optional[Image.Image], str]:
        """Generate image using Gemini with optional reference images"""
        if not self.api_key:
            return None, "❌ API key not configured"
        
        # Try different image models as fallback
        last_error = None
        for model_name in POSSIBLE_IMAGE_MODELS:
            try:
                model = genai.GenerativeModel(model_name)
                
                # Prepare content for generation
                content_parts = []
                
                # Add reference images if provided
                if reference_images:
                    for ref_img in reference_images:
                        content_parts.append(ref_img)
                    content_parts.append(f"Using the reference images above as style guide, generate: {prompt}")
                else:
                    content_parts.append(f"Generate an image: {prompt}")
                
                response = model.generate_content(content_parts)
                
                # Extract image from response - enhanced error handling
                parts_found = 0
                inline_data_found = 0
                
                # Debug: Check response structure
                if hasattr(response, '_result') and hasattr(response._result, 'candidates'):
                    for candidate in response._result.candidates:
                        if hasattr(candidate, 'content') and hasattr(candidate.content, 'parts'):
                            for part in candidate.content.parts:
                                parts_found += 1
                                
                                # Check for inline_data attribute
                                if hasattr(part, 'inline_data') and part.inline_data:
                                    inline_data_found += 1
                                    try:
                                        # Get the raw data
                                        inline_data = part.inline_data
                                        
                                        # Check if data is already bytes or needs decoding
                                        if hasattr(inline_data, 'data'):
                                            image_data = inline_data.data
                                            
                                            # If it's string, base64 decode it
                                            if isinstance(image_data, str):
                                                image_data = base64.b64decode(image_data)
                                            # If it's already bytes, use directly
                                            elif isinstance(image_data, bytes):
                                                pass
                                            else:
                                                continue  # Try next model
                                            
                                            # Create PIL Image from bytes
                                            image_buffer = BytesIO(image_data)
                                            image = Image.open(image_buffer)
                                            
                                            # Verify the image is valid by loading it
                                            image.load()
                                            
                                            return image, f"✅ Image generated successfully with {model_name}!"
                                            
                                    except Exception as decode_error:
                                        last_error = f"Error decoding image with {model_name}: {decode_error}"
                                        continue  # Try next model
                                
                                # Check for other possible image data locations
                                elif hasattr(part, 'text') and part.text:
                                    # Sometimes image data might be in text as base64
                                    text = part.text.strip()
                                    if text.startswith('data:image/') or len(text) > 1000:  # Might be base64 image data
                                        try:
                                            # Try to decode as base64
                                            if text.startswith('data:image/'):
                                                # Remove data URL prefix
                                                text = text.split(',', 1)[1] if ',' in text else text
                                            
                                            image_data = base64.b64decode(text)
                                            image_buffer = BytesIO(image_data)
                                            image = Image.open(image_buffer)
                                            image.load()
                                            
                                            return image, f"✅ Image generated successfully with {model_name} (from text data)!"
                                            
                                        except Exception:
                                            continue  # Try next model
                
                # No image found in this model's response
                response_text = getattr(response, 'text', 'No response text available')
                last_error = f"Model {model_name}: No image in response. Response: {response_text[:100]}..."
                
            except Exception as e:
                error_msg = str(e)
                last_error = f"Model {model_name} failed: {error_msg}"
                continue  # Try next model
        
        # If we get here, all models failed
        if last_error:
            if "404" in last_error or "not found" in last_error.lower():
                return None, f"❌ No image models accessible. Tried: {', '.join(POSSIBLE_IMAGE_MODELS)}\n\nThis may be due to:\n• Models not available in your region\n• Insufficient API access level\n• Model names changed\n\nLast error: {last_error}"
            elif "400" in last_error or "invalid" in last_error.lower():
                return None, f"❌ API request error: {last_error}"
            else:
                return None, f"❌ All image models failed. Tried: {', '.join(POSSIBLE_IMAGE_MODELS)}\n\nLast error: {last_error}"
        else:
            return None, f"❌ No image models could be initialized. Tried: {', '.join(POSSIBLE_IMAGE_MODELS)}"
    
    def save_generated_image(self, image: Image.Image, directory: Path, prefix: str = "gen") -> Path:
        """Save generated image to specified directory"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}_{uuid.uuid4().hex[:6]}.png"
        filepath = directory / filename
        image.save(filepath)
        return filepath

# =============================================================================
# ELEVENLABS VOICE SYSTEM
# =============================================================================

class ElevenLabsClient:
    """ElevenLabs API client for voice generation"""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.session = requests.Session()
        if api_key:
            self.session.headers.update({"xi-api-key": api_key})
    
    def configure_api(self, api_key: str) -> Tuple[bool, str]:
        """Configure ElevenLabs API key and test connection"""
        self.api_key = api_key
        self.session.headers.update({"xi-api-key": api_key})
        
        try:
            # Test API access by getting available voices
            response = self.session.get(f"{ELEVENLABS_API_BASE}/voices")
            
            if response.status_code == 200:
                voices_data = response.json()
                voice_count = len(voices_data.get("voices", []))
                return True, f"✅ ElevenLabs API connected! Found {voice_count} available voices."
            elif response.status_code == 401:
                return False, "❌ Invalid API key. Please check your ElevenLabs API key."
            else:
                return False, f"❌ API test failed with status {response.status_code}: {response.text}"
                
        except requests.exceptions.RequestException as e:
            return False, f"❌ Connection error: {str(e)}"
        except Exception as e:
            return False, f"❌ Unexpected error: {str(e)}"
    
    def get_available_voices(self) -> Tuple[bool, List[Dict[str, Any]]]:
        """Get list of available voices from ElevenLabs"""
        if not self.api_key:
            return False, []
        
        try:
            response = self.session.get(f"{ELEVENLABS_API_BASE}/voices")
            
            if response.status_code == 200:
                voices_data = response.json()
                voices = voices_data.get("voices", [])
                return True, voices
            else:
                print(f"Failed to get voices: {response.status_code} - {response.text}")
                return False, []
                
        except Exception as e:
            print(f"Error getting voices: {e}")
            return False, []
    
    def generate_speech(self, text: str, voice_id: str, settings: Dict[str, Any] = None) -> Tuple[bool, bytes, str]:
        """Generate speech from text using specified voice"""
        if not self.api_key:
            return False, b"", "API key not configured"
        
        if not settings:
            settings = DEFAULT_VOICE_SETTINGS.copy()
        
        url = f"{ELEVENLABS_API_BASE}/text-to-speech/{voice_id}"
        
        payload = {
            "text": text,
            "model_id": "eleven_monolingual_v1",
            "voice_settings": settings
        }
        
        try:
            response = self.session.post(url, json=payload, stream=True)
            
            if response.status_code == 200:
                audio_data = b""
                for chunk in response.iter_content(chunk_size=1024):
                    audio_data += chunk
                return True, audio_data, "Speech generated successfully"
            else:
                error_msg = f"Speech generation failed: {response.status_code} - {response.text}"
                return False, b"", error_msg
                
        except Exception as e:
            return False, b"", f"Error generating speech: {str(e)}"

class ElevenLabsVoiceStudio:
    """Main voice studio for character audio generation"""
    
    def __init__(self, api_key: str = ""):
        self.client = ElevenLabsClient(api_key)
        self.voice_profiles: Dict[str, VoiceProfile] = {}
        self.audio_scenes: List[AudioScene] = []
        
        # Load existing voice profiles if any
        self.load_voice_profiles()
    
    def configure_api(self, api_key: str) -> str:
        """Configure API key"""
        success, message = self.client.configure_api(api_key)
        return message
    
    def save_voice_profiles(self):
        """Save voice profiles to file"""
        profiles_file = VOICE_DIR / "voice_profiles.json"
        profiles_data = {
            profile_id: {
                "id": profile.id,
                "character_name": profile.character_name,
                "voice_id": profile.voice_id,
                "voice_name": profile.voice_name,
                "settings": profile.settings,
                "created_at": profile.created_at
            }
            for profile_id, profile in self.voice_profiles.items()
        }
        
        with open(profiles_file, 'w') as f:
            json.dump(profiles_data, f, indent=2)
    
    def load_voice_profiles(self):
        """Load voice profiles from file"""
        profiles_file = VOICE_DIR / "voice_profiles.json"
        if profiles_file.exists():
            try:
                with open(profiles_file, 'r') as f:
                    profiles_data = json.load(f)
                
                for profile_id, data in profiles_data.items():
                    profile = VoiceProfile(
                        id=data["id"],
                        character_name=data["character_name"],
                        voice_id=data["voice_id"],
                        voice_name=data["voice_name"],
                        settings=data["settings"],
                        created_at=data["created_at"]
                    )
                    self.voice_profiles[profile_id] = profile
            except Exception as e:
                print(f"Error loading voice profiles: {e}")
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available voices for selection"""
        success, voices = self.client.get_available_voices()
        if success:
            return voices
        return []
    
    def create_voice_profile(self, character_name: str, voice_id: str, voice_name: str, settings: Dict[str, Any] = None) -> str:
        """Create a voice profile for a character"""
        if not settings:
            settings = DEFAULT_VOICE_SETTINGS.copy()
        
        profile_id = uuid.uuid4().hex[:12]
        profile = VoiceProfile(
            id=profile_id,
            character_name=character_name,
            voice_id=voice_id,
            voice_name=voice_name,
            settings=settings
        )
        
        self.voice_profiles[profile_id] = profile
        self.save_voice_profiles()
        
        return f"✅ Voice profile created for {character_name} using {voice_name}"
    
    def generate_character_dialogue(self, character_name: str, dialogue_text: str) -> Tuple[bool, Optional[Path], str]:
        """Generate dialogue audio for a character"""
        # Find voice profile for character
        character_profile = None
        for profile in self.voice_profiles.values():
            if profile.character_name.lower() == character_name.lower():
                character_profile = profile
                break
        
        if not character_profile:
            return False, None, f"❌ No voice profile found for character '{character_name}'"
        
        # Generate speech
        success, audio_data, message = self.client.generate_speech(
            dialogue_text,
            character_profile.voice_id,
            character_profile.settings
        )
        
        if not success:
            return False, None, f"❌ {message}"
        
        # Save audio file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{character_name.lower().replace(' ', '_')}_{timestamp}_{uuid.uuid4().hex[:6]}.mp3"
        audio_path = DIALOGUE_DIR / filename
        
        with open(audio_path, 'wb') as f:
            f.write(audio_data)
        
        return True, audio_path, f"✅ Generated dialogue for {character_name}: {len(dialogue_text)} characters"
    
    def generate_scene_audio(self, scene_description: str, dialogue_lines: List[Dict[str, str]]) -> Tuple[bool, Optional[AudioScene], str]:
        """Generate audio for an entire scene with multiple characters"""
        if not dialogue_lines:
            return False, None, "❌ No dialogue lines provided"
        
        scene_id = uuid.uuid4().hex[:8]
        audio_scene = AudioScene(
            id=scene_id,
            scene_description=scene_description,
            characters=[],
            dialogue_lines=[]
        )
        
        # Generate audio for each dialogue line
        for line in dialogue_lines:
            character = line.get("character", "")
            text = line.get("text", "")
            
            if not character or not text:
                continue
            
            success, audio_path, message = self.generate_character_dialogue(character, text)
            
            if success:
                audio_scene.characters.append(character)
                audio_scene.dialogue_lines.append({
                    "character": character,
                    "text": text,
                    "audio_file": str(audio_path)
                })
            else:
                return False, None, f"❌ Failed to generate audio for {character}: {message}"
        
        self.audio_scenes.append(audio_scene)
        return True, audio_scene, f"✅ Generated scene audio with {len(audio_scene.dialogue_lines)} dialogue lines"
    
    def generate_voiceover(self, text: str, voice_id: str = None, voice_name: str = "Narrator") -> Tuple[bool, Optional[Path], str]:
        """Generate narrator/voiceover audio"""
        if not voice_id:
            # Use a default narrative voice if available
            voices = self.get_available_voices()
            if voices:
                # Look for a good narrative voice
                for voice in voices:
                    if any(keyword in voice.get("name", "").lower() for keyword in ["narrator", "story", "documentary", "news"]):
                        voice_id = voice["voice_id"]
                        break
                
                # Fallback to first available voice
                if not voice_id:
                    voice_id = voices[0]["voice_id"]
            else:
                return False, None, "❌ No voices available"
        
        success, audio_data, message = self.client.generate_speech(text, voice_id)
        
        if not success:
            return False, None, f"❌ {message}"
        
        # Save voiceover file
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"voiceover_{timestamp}_{uuid.uuid4().hex[:6]}.mp3"
        audio_path = DIALOGUE_DIR / filename
        
        with open(audio_path, 'wb') as f:
            f.write(audio_data)
        
        return True, audio_path, f"✅ Generated voiceover: {len(text)} characters"
    
    def test_api_connection(self) -> str:
        """Test API connection and return status"""
        return self.configure_api(self.client.api_key)

# =============================================================================
# CHARACTER DEVELOPMENT SYSTEM
# =============================================================================

class CharacterDeveloper:
    """AI-powered character development from v6.py"""
    
    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.conversation_history = []
    
    def develop_character_chat(self, user_input: str, chat_history: List) -> Tuple[List, str, str]:
        """Interactive character development chat"""
        if not self.api_key:
            return chat_history, "", "❌ Please configure API key first"
        
        try:
            chat_history.append({"role": "user", "content": user_input})
            
            system_prompt = """You are an expert character designer and storyteller. Help the user create detailed, consistent characters for comic strips and storyboards.

Focus on:
- Physical appearance (height, build, hair, eyes, clothing style)
- Personality traits and quirks  
- Background story and history
- Distinctive features that make the character recognizable
- How the character would look from different angles and scenarios

Be conversational and ask follow-up questions to refine the character. When the character description seems complete, provide a detailed visual description suitable for image generation and suggest the user save the character to their library."""
            
            # Build context from recent chat history
            context = ""
            if len(chat_history) > 1:
                recent_messages = chat_history[-10:]  # Last 10 messages
                context_parts = []
                for msg in recent_messages[:-1]:  # Exclude the current one
                    if isinstance(msg, dict) and msg.get("content"):
                        role = msg.get("role", "user").title()
                        context_parts.append(f"{role}: {msg['content']}")
                context = "\n".join(context_parts)
            
            full_prompt = f"{system_prompt}\n\nRecent conversation:\n{context}\n\nUser: {user_input}"
            
            model = genai.GenerativeModel(TEXT_MODEL)
            response = model.generate_content(full_prompt)
            
            ai_response = response.text
            chat_history.append({"role": "assistant", "content": ai_response})
            
            # Check if character seems ready
            status_message = ""
            ready_keywords = ["character is ready", "ready for image generation", "save to library", "complete description"]
            if any(keyword in ai_response.lower() for keyword in ready_keywords):
                status_message = "✅ Character development complete! Ready to save to library."
            
            return chat_history, "", status_message
            
        except Exception as e:
            error_msg = f"❌ Error: {str(e)}"
            chat_history[-1][1] = error_msg
            return chat_history, "", error_msg
    
    def extract_character_from_chat(self, chat_history: List) -> Dict[str, str]:
        """Extract character details from chat history using AI"""
        if not self.api_key or not chat_history:
            # Return basic extraction from chat without AI
            return self._basic_chat_extraction(chat_history)
        
        try:
            # Combine chat history into text
            conversation_text = ""
            for msg in chat_history:
                if isinstance(msg, dict) and msg.get("content"):
                    role = msg.get("role", "user").title()
                    conversation_text += f"{role}: {msg['content']}\n\n"
            
            if not conversation_text.strip():
                return self._basic_chat_extraction(chat_history)
            
            extraction_prompt = f"""Based on the following character development conversation, extract and organize the key character information:

{conversation_text}

Please provide a structured summary in this exact format:
CHARACTER_NAME: [name if mentioned, or "Unknown"]
PHYSICAL_DESCRIPTION: [appearance, clothing, distinctive features]
PERSONALITY: [traits, quirks, behavior patterns]
BACKGROUND: [history, profession, relationships]
STYLE_NOTES: [artistic style, visual direction]

Be concise but capture the key details discussed."""
            
            model = genai.GenerativeModel(TEXT_MODEL)
            response = model.generate_content(extraction_prompt)
            
            # Parse response into components
            text = response.text
            extracted = {
                "name": "",
                "description": "",
                "personality": "",
                "background": "",
                "style_notes": ""
            }
            
            # Enhanced parsing
            lines = text.split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('CHARACTER_NAME:'):
                    name = line.split(':', 1)[1].strip()
                    if name.lower() != "unknown":
                        extracted["name"] = name
                elif line.startswith('PHYSICAL_DESCRIPTION:'):
                    extracted["description"] = line.split(':', 1)[1].strip()
                elif line.startswith('PERSONALITY:'):
                    extracted["personality"] = line.split(':', 1)[1].strip()
                elif line.startswith('BACKGROUND:'):
                    extracted["background"] = line.split(':', 1)[1].strip()
                elif line.startswith('STYLE_NOTES:'):
                    extracted["style_notes"] = line.split(':', 1)[1].strip()
            
            # Fallback to basic extraction if AI extraction failed
            if not any(extracted.values()):
                return self._basic_chat_extraction(chat_history)
            
            return extracted
            
        except Exception as e:
            print(f"Error extracting character with AI: {e}")
            return self._basic_chat_extraction(chat_history)
    
    def _basic_chat_extraction(self, chat_history: List) -> Dict[str, str]:
        """Basic character extraction without AI"""
        if not chat_history:
            return {}
        
        # Combine all conversation text
        full_text = ""
        for msg in chat_history:
            if isinstance(msg, dict) and msg.get("content"):
                role = msg.get("role", "user").title()
                full_text += f"{role}: {msg['content']}\n"
        
        # Try to find character name using simple patterns
        name = ""
        text_lower = full_text.lower()
        name_patterns = [
            "character named ",
            "character called ",
            "name is ",
            "named ",
            "called "
        ]
        
        for pattern in name_patterns:
            if pattern in text_lower:
                start_idx = text_lower.find(pattern) + len(pattern)
                # Get next few words
                remaining = full_text[start_idx:start_idx+50]
                words = remaining.split()
                if words:
                    potential_name = words[0].strip(".,!?\"'()[]")
                    if potential_name.isalpha() and len(potential_name) > 1:
                        name = potential_name.capitalize()
                        break
        
        return {
            "name": name,
            "description": f"Character developed through conversation. Full chat:\n\n{full_text[:500]}...",
            "personality": "",
            "background": "",
            "style_notes": "Character created from chat - refine details in Character Library"
        }

# =============================================================================
# MAIN APPLICATION CLASS
# =============================================================================

class ComprehensiveStudio:
    """Main application class combining all systems"""
    
    def __init__(self):
        self.data_manager = DataManager()
        self.image_generator = ImageGenerator(self.data_manager.config.api_key)
        self.character_developer = CharacterDeveloper(self.data_manager.config.api_key)
        self.voice_studio = ElevenLabsVoiceStudio(self.data_manager.config.elevenlabs_api_key)
        
        # Current state
        self.current_project: Optional[Project] = None
        self.selected_character_id: Optional[str] = None
        self.temp_generated_image: Optional[Image.Image] = None
        self.temp_image_path: Optional[Path] = None
        
        # Load last project if available
        if self.data_manager.config.last_project:
            self.current_project = self.data_manager.load_project(self.data_manager.config.last_project)
    
    def configure_api_key(self, api_key: str) -> str:
        """Configure API key for all systems"""
        success, message = self.image_generator.configure_api(api_key)
        if success:
            self.data_manager.config.api_key = api_key
            self.character_developer.api_key = api_key
            self.data_manager.save_config(self.data_manager.config)
        return message
    
    def create_project(self, name: str, description: str = "") -> Tuple[str, str]:
        """Create new project"""
        if not name.strip():
            return "", "❌ Project name required"
        
        # Check if project already exists
        existing_projects = self.data_manager.list_projects()
        if any(p.name.lower() == name.lower() for p in existing_projects):
            return "", f"❌ Project '{name}' already exists"
        
        project = Project(
            name=name.strip(),
            id=uuid.uuid4().hex[:12],
            description=description.strip()
        )
        
        self.data_manager.save_project(project)
        self.current_project = project
        self.data_manager.config.last_project = project.id
        self.data_manager.save_config(self.data_manager.config)
        
        return project.id, f"✅ Created project '{name}'"
    
    def load_project(self, project_id: str) -> str:
        """Load existing project"""
        project = self.data_manager.load_project(project_id)
        if project:
            self.current_project = project
            self.data_manager.config.last_project = project_id
            self.data_manager.save_config(self.data_manager.config)
            return f"✅ Loaded project '{project.name}'"
        return "❌ Project not found"
    
    def get_project_choices(self) -> List[Tuple[str, str]]:
        """Get project choices for dropdown (name, id)"""
        projects = self.data_manager.list_projects()
        return [(f"{p.name} (Updated: {p.updated_at[:10]})", p.id) for p in projects]
    
    def save_character_from_chat(self, chat_history: List, name_override: str = "") -> str:
        """Save character developed through chat"""
        if not chat_history:
            return "❌ No chat history to save. Please use the Character Library tab to create characters manually."
        
        # Extract character information from chat
        extracted = self.character_developer.extract_character_from_chat(chat_history)
        
        # Use override name if provided, otherwise try to extract from chat
        char_name = name_override.strip()
        if not char_name and extracted:
            char_name = extracted.get("name", "").strip()
        
        # If still no name, try to get it from the first few messages
        if not char_name and chat_history:
            for exchange in chat_history[:3]:  # Check first 3 exchanges
                try:
                    # Handle different chat history formats
                    if isinstance(exchange, (list, tuple)) and len(exchange) >= 2:
                        user_msg, assistant_msg = exchange[0], exchange[1]
                    elif isinstance(exchange, dict):
                        user_msg = exchange.get('user', '') or exchange.get(0, '')
                        assistant_msg = exchange.get('assistant', '') or exchange.get(1, '')
                    else:
                        continue
                    
                    if user_msg and assistant_msg:  # Both user and assistant messages exist
                        # Look for character names in the conversation
                        user_text = str(user_msg).lower()
                        assistant_text = str(assistant_msg).lower()
                    
                        # Simple heuristics to find character names
                        if "character named" in user_text or "character called" in user_text:
                            words = str(user_msg).split()
                            for i, word in enumerate(words):
                                if word.lower() in ["named", "called"] and i + 1 < len(words):
                                    potential_name = words[i + 1].strip(".,!?\"'")
                                    if potential_name.isalpha():
                                        char_name = potential_name
                                        break
                        
                        if char_name:
                            break
                            
                except (IndexError, KeyError, TypeError):
                    # Skip malformed exchanges
                    continue
        
        # If we still don't have a name, require user to provide one
        if not char_name:
            return "❌ Character name required. Please enter a name in the 'Character Name' field above."
        
        # Build character description from chat if extraction worked
        description = ""
        personality = ""
        background = ""
        style_notes = ""
        
        if extracted:
            description = extracted.get("description", "").strip()
            personality = extracted.get("personality", "").strip()
            background = extracted.get("background", "").strip()
            style_notes = extracted.get("style_notes", "").strip()
        
        # If extraction failed, create a basic description from chat
        if not description and chat_history:
            chat_summary = []
            for exchange in chat_history:
                if exchange[0] and exchange[1]:
                    chat_summary.append(f"User: {exchange[0]}")
                    chat_summary.append(f"AI: {exchange[1]}")
            
            if chat_summary:
                description = f"Character developed through chat conversation:\n\n" + "\n".join(chat_summary[-6:])  # Last 3 exchanges
                style_notes = "Character created from AI conversation - refine details in Character Library"
        
        # Create character with available information
        character = Character(
            id=uuid.uuid4().hex[:12],
            name=char_name,
            description=description or "Character created from chat - add details in Character Library",
            personality=personality,
            background=background,
            style_notes=style_notes
        )
        
        self.data_manager.save_character(character)
        return f"✅ Saved character '{char_name}' to library! You can refine details in the Character Library tab."

    def get_character_choices(self) -> List[Tuple[str, str]]:
        """Get character choices for dropdown (name, id)"""
        characters = self.data_manager.list_characters()
        return [(char.name, char.id) for char in characters]
    
    def load_character_details(self, char_id: str) -> Tuple[str, str, str, str, str, List[str]]:
        """Load character details for editing"""
        character = self.data_manager.load_character(char_id)
        if character:
            self.selected_character_id = char_id
            gallery_paths = [str(path) for path in character.get_reference_paths()]
            return (character.name, character.description, character.personality, 
                    character.background, character.style_notes, gallery_paths)
        return "", "", "", "", "", []
    
    def save_character_details(self, char_id: str, name: str, description: str, 
                              personality: str, background: str, style_notes: str) -> str:
        """Save updated character details"""
        if not name.strip():
            return "❌ Character name required"
        
        if char_id:
            # Update existing character
            character = self.data_manager.load_character(char_id)
            if not character:
                return "❌ Character not found"
        else:
            # Create new character
            character = Character(
                id=uuid.uuid4().hex[:12], 
                name="", 
                description=""
            )
        
        character.name = name.strip()
        character.description = description.strip()
        character.personality = personality.strip()
        character.background = background.strip()
        character.style_notes = style_notes.strip()
        
        self.data_manager.save_character(character)
        self.selected_character_id = character.id
        return f"✅ Saved character '{character.name}'"
    
    def add_reference_image(self, char_id: str, image: Image.Image) -> Tuple[str, List[str]]:
        """Add reference image to character"""
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found", []
        
        # Save image to character's reference directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ref_{timestamp}_{uuid.uuid4().hex[:6]}.png"
        image_path = character.references_dir / filename
        image.save(image_path)
        
        # Update character metadata
        character.references.append(filename)
        self.data_manager.save_character(character)
        
        gallery_paths = [str(path) for path in character.get_reference_paths()]
        return f"✅ Added reference image", gallery_paths
    
    def generate_character_image(self, char_id: str, prompt_modifier: str, use_references: bool = True) -> Tuple[str, Optional[Image.Image]]:
        """Generate image for character"""
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found", None
        
        # Build comprehensive prompt
        prompt_parts = [
            f"Character: {character.name}",
            f"Description: {character.description}",
        ]
        if character.personality:
            prompt_parts.append(f"Personality: {character.personality}")
        if character.style_notes:
            prompt_parts.append(f"Style: {character.style_notes}")
        if prompt_modifier.strip():
            prompt_parts.append(f"Specific request: {prompt_modifier.strip()}")
        
        full_prompt = "\n".join(prompt_parts)
        
        # Get reference images if requested
        reference_images = []
        if use_references:
            # Use project-specific references if available, fallback to global character references
            project_ref_paths = [Path(p) for p in self.get_project_character_references(char_id)]
            global_ref_paths = character.get_reference_paths()
            
            # Prioritize project references, then global ones
            all_ref_paths = project_ref_paths + global_ref_paths
            
            for ref_path in all_ref_paths[:3]:  # Limit to 3 references
                try:
                    ref_img = Image.open(ref_path)
                    reference_images.append(ref_img)
                except Exception:
                    continue
        
        # Generate image
        generated_image, message = self.image_generator.generate_image(full_prompt, reference_images)
        
        if generated_image:
            self.temp_generated_image = generated_image
            # Save temporary copy
            self.temp_image_path = self.image_generator.save_generated_image(
                generated_image, SCENES_DIR, "temp"
            )
        
        return message, generated_image
    
    def add_reference_to_project_character(self, char_id: str, image: Image.Image, description: str = "") -> Tuple[str, List[str]]:
        """Add reference image to character in current project"""
        if not self.current_project:
            return "❌ No current project", []
        
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found", []
        
        # Save to project-specific character directory
        project_char_dir = get_project_character_dir(self.current_project.id, char_id)
        ref_dir = project_char_dir / "references"
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"ref_{timestamp}_{uuid.uuid4().hex[:6]}.png"
        image_path = ref_dir / filename
        
        # Save image
        image.save(image_path)
        
        # Load or create project character references list
        refs_file = project_char_dir / "references.json"
        if refs_file.exists():
            with open(refs_file, 'r') as f:
                references = json.load(f)
        else:
            references = []
        
        # Add new reference
        ref_entry = {
            "filename": filename,
            "description": description,
            "created_at": datetime.now().isoformat()
        }
        references.append(ref_entry)
        
        # Save references list
        with open(refs_file, 'w') as f:
            json.dump(references, f, indent=2)
        
        # Get updated gallery paths
        gallery_paths = self.get_project_character_references(char_id)
        
        return f"✅ Added reference image to {character.name}", gallery_paths
    
    def get_project_character_references(self, char_id: str) -> List[str]:
        """Get all reference images for character in current project"""
        if not self.current_project:
            return []
        
        project_char_dir = get_project_character_dir(self.current_project.id, char_id)
        refs_file = project_char_dir / "references.json"
        
        if not refs_file.exists():
            return []
        
        try:
            with open(refs_file, 'r') as f:
                references = json.load(f)
            
            gallery_paths = []
            ref_dir = project_char_dir / "references"
            for ref in references:
                ref_path = ref_dir / ref["filename"]
                if ref_path.exists():
                    gallery_paths.append(str(ref_path))
            
            return gallery_paths
        except Exception as e:
            print(f"Error loading references: {e}")
            return []

    def save_generated_to_references(self, char_id: str, description: str = "") -> Tuple[str, List[str]]:
        """Save temporary generated image as character reference in current project"""
        if not self.temp_generated_image:
            return "❌ No generated image to save", []
        
        result, gallery_paths = self.add_reference_to_project_character(
            char_id, self.temp_generated_image, description or "Generated image"
        )
        
        # Clean up temp image
        if self.temp_image_path and self.temp_image_path.exists():
            self.temp_image_path.unlink()
        self.temp_generated_image = None
        self.temp_image_path = None
        
        return result, gallery_paths
    
    def delete_reference_from_project_character(self, char_id: str, ref_indices: List[int]) -> Tuple[str, List[str]]:
        """Delete selected reference images from character in current project"""
        if not self.current_project:
            return "❌ No current project", []
        
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found", []
        
        try:
            project_char_dir = get_project_character_dir(self.current_project.id, char_id)
            refs_file = project_char_dir / "references.json"
            
            if not refs_file.exists():
                return "❌ No references to delete", []
            
            # Load current references
            with open(refs_file, 'r') as f:
                references = json.load(f)
            
            if not references:
                return "❌ No references to delete", []
            
            # Sort indices in reverse order to avoid index shifting issues
            ref_indices.sort(reverse=True)
            deleted_count = 0
            
            for idx in ref_indices:
                if 0 <= idx < len(references):
                    ref_to_delete = references[idx]
                    ref_file_path = project_char_dir / "references" / ref_to_delete["filename"]
                    
                    # Delete the physical file
                    if ref_file_path.exists():
                        ref_file_path.unlink()
                    
                    # Remove from references list
                    references.pop(idx)
                    deleted_count += 1
            
            # Save updated references
            with open(refs_file, 'w') as f:
                json.dump(references, f, indent=2)
            
            # Get updated gallery paths
            gallery_paths = []
            ref_dir = project_char_dir / "references"
            for ref in references:
                ref_path = ref_dir / ref["filename"]
                if ref_path.exists():
                    gallery_paths.append(str(ref_path))
            
            return f"✅ Deleted {deleted_count} reference(s)", gallery_paths
            
        except Exception as e:
            return f"❌ Error deleting references: {str(e)}", []
    
    def expand_image(self, base_image: Image.Image, direction: str, expansion_amount: str, instruction: str = "") -> Tuple[str, Optional[Image.Image]]:
        """Expand/outpaint image in specified direction for camera angle changes"""
        try:
            # Create expansion canvas based on direction and amount
            original_width, original_height = base_image.size
            
            # Parse expansion amount (pixels or percentage)
            if expansion_amount.endswith('%'):
                percentage = float(expansion_amount[:-1]) / 100
                if direction in ['left', 'right']:
                    expansion_px = int(original_width * percentage)
                else:  # up, down
                    expansion_px = int(original_height * percentage)
            else:
                expansion_px = int(expansion_amount)
            
            # Create new canvas with expanded dimensions
            if direction == 'right':
                new_width = original_width + expansion_px
                new_height = original_height
                paste_position = (0, 0)
                expansion_area = (original_width, 0, new_width, new_height)
            elif direction == 'left':
                new_width = original_width + expansion_px
                new_height = original_height
                paste_position = (expansion_px, 0)
                expansion_area = (0, 0, expansion_px, new_height)
            elif direction == 'up':
                new_width = original_width
                new_height = original_height + expansion_px
                paste_position = (0, expansion_px)
                expansion_area = (0, 0, new_width, expansion_px)
            elif direction == 'down':
                new_width = original_width
                new_height = original_height + expansion_px
                paste_position = (0, 0)
                expansion_area = (0, original_height, new_width, new_height)
            else:
                return "❌ Invalid direction. Use: left, right, up, down", None
            
            # Create expanded canvas (white background)
            expanded_canvas = Image.new('RGB', (new_width, new_height), (255, 255, 255))
            expanded_canvas.paste(base_image, paste_position)
            
            # Build prompt for outpainting
            base_instruction = instruction if instruction else f"Expand the image to the {direction} showing more of the scene from a wider camera angle"
            
            outpaint_prompt = f"""Expand this image to the {direction} by filling in the white/empty area.

{base_instruction}

Maintain consistent lighting, style, and perspective. The expansion should look natural and seamless, as if the camera pulled back or moved to show more of the scene. Keep the same character appearances and environmental details."""
            
            # Generate the expanded image
            generated_image, status = self.image_generator.generate_image(outpaint_prompt, [expanded_canvas])
            
            if generated_image:
                self.temp_generated_image = generated_image
                return f"✅ Image expanded {direction} by {expansion_amount}", generated_image
            else:
                return f"❌ {status}", None
                
        except Exception as e:
            return f"❌ Error expanding image: {str(e)}", None
    
    def chat_generate_image(self, char_id: str, user_instruction: str) -> Tuple[str, Optional[Image.Image], List[Tuple[str, str]]]:
        """Chat-based iterative image generation for character"""
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found", None, []
        
        # Add user message to chat history
        character.chat_history.append(("user", user_instruction))
        
        try:
            # Build the prompt with character header and instruction
            prompt_header = character.get_prompt_header()
            full_prompt = f"{prompt_header}\n\nUser instruction: {user_instruction}"
            
            # Get reference images (up to 3 most recent)
            ref_images = []
            if character.references:
                recent_refs = character.references[-3:]  # Last 3 references for consistency
                for ref_name in recent_refs:
                    ref_path = character.references_dir / ref_name
                    if ref_path.exists():
                        try:
                            ref_img = Image.open(ref_path).convert("RGB")
                            ref_images.append(ref_img)
                        except Exception:
                            pass
            
            # Generate image
            generated_image, status = self.image_generator.generate_image(full_prompt, ref_images)
            
            if generated_image:
                # Store temporarily for potential saving to references
                self.temp_generated_image = generated_image
                
                # Add assistant response to chat history
                assistant_response = f"Generated image based on: {user_instruction}"
                character.chat_history.append(("assistant", assistant_response))
                
                # Save updated character with chat history
                self.data_manager.save_character(character)
                
                return f"✅ {status}", generated_image, character.chat_history
            else:
                character.chat_history.append(("assistant", f"Failed to generate image: {status}"))
                self.data_manager.save_character(character)
                return f"❌ {status}", None, character.chat_history
        
        except Exception as e:
            error_msg = f"Error generating image: {str(e)}"
            character.chat_history.append(("assistant", error_msg))
            self.data_manager.save_character(character)
            return f"❌ {error_msg}", None, character.chat_history
    
    def generate_with_selected_references(self, char_id: str, prompt: str, selected_indices: List[int]) -> Tuple[str, Optional[Image.Image]]:
        """Generate image using specific selected reference images"""
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found", None
        
        try:
            # Get selected reference images
            ref_images = []
            project_refs = self.get_project_character_references(char_id)
            
            for idx in selected_indices:
                if 0 <= idx < len(project_refs):
                    try:
                        ref_img = Image.open(project_refs[idx]).convert("RGB")
                        ref_images.append(ref_img)
                    except Exception:
                        pass
            
            # Build prompt with character info
            full_prompt = f"{character.get_prompt_header()}\n\nGeneration instruction: {prompt}"
            
            # Generate image
            generated_image, status = self.image_generator.generate_image(full_prompt, ref_images)
            
            if generated_image:
                self.temp_generated_image = generated_image
                return f"✅ Generated with {len(ref_images)} selected references", generated_image
            else:
                return f"❌ {status}", None
                
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def generate_with_single_reference(self, char_id: str, prompt: str, ref_index: int) -> Tuple[str, Optional[Image.Image]]:
        """Generate image for iterative changes using single reference"""
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found", None
        
        try:
            # Get single reference image
            project_refs = self.get_project_character_references(char_id)
            
            if not (0 <= ref_index < len(project_refs)):
                return "❌ Invalid reference selection", None
            
            ref_image = Image.open(project_refs[ref_index]).convert("RGB")
            
            # Build prompt for iterative changes
            iterative_prompt = f"Based on this reference image, {prompt}. Maintain the same character identity, style, and overall composition while making the requested changes."
            
            # Generate image
            generated_image, status = self.image_generator.generate_image(iterative_prompt, [ref_image])
            
            if generated_image:
                self.temp_generated_image = generated_image
                return f"✅ Generated iterative change from reference", generated_image
            else:
                return f"❌ {status}", None
                
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def generate_scene_with_selected_references(self, character_ids: List[str], scene_instruction: str, selected_ref_indices: List[int]) -> Tuple[str, Optional[Image.Image]]:
        """Generate scene using specific selected reference images from all characters"""
        try:
            # Get all character references and build selected reference list
            all_refs = []
            ref_to_char_map = {}
            current_index = 0
            
            for char_id in character_ids:
                char_refs = self.get_project_character_references(char_id)
                for ref_path in char_refs:
                    all_refs.append(ref_path)
                    ref_to_char_map[current_index] = char_id
                    current_index += 1
            
            # Get selected reference images
            selected_ref_images = []
            for idx in selected_ref_indices:
                if 0 <= idx < len(all_refs):
                    try:
                        ref_img = Image.open(all_refs[idx]).convert("RGB")
                        selected_ref_images.append(ref_img)
                    except Exception:
                        pass
            
            # Load character data for prompt building
            characters = []
            for char_id in character_ids:
                character = self.data_manager.load_character(char_id)
                if character:
                    characters.append(character)
            
            if not characters:
                return "❌ No valid characters found", None
            
            # Build enhanced scene prompt
            prompt_parts = [f"Multi-character scene: {scene_instruction}"]
            prompt_parts.append("\nCharacters in scene:")
            
            for char in characters:
                char_desc = f"- {char.name}: {char.description}"
                if char.personality:
                    char_desc += f" (Personality: {char.personality})"
                prompt_parts.append(char_desc)
            
            # Add style consistency
            styles = [char.style_notes for char in characters if char.style_notes]
            if styles:
                dominant_style = max(set(styles), key=styles.count) if styles else "photorealistic"
                prompt_parts.append(f"\nMaintain consistent {dominant_style} style throughout the scene.")
            
            prompt_parts.append("Ensure all characters are consistent with their individual descriptions while maintaining scene cohesion.")
            full_prompt = "\n".join(prompt_parts)
            
            # Generate scene with selected references
            generated_image, status = self.image_generator.generate_image(full_prompt, selected_ref_images)
            
            if generated_image:
                self.temp_generated_image = generated_image
                return f"✅ Scene generated with {len(selected_ref_images)} selected references", generated_image
            else:
                return f"❌ {status}", None
                
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def generate_scene_with_single_reference(self, character_ids: List[str], scene_instruction: str, ref_index: int) -> Tuple[str, Optional[Image.Image]]:
        """Generate scene variations using single reference"""
        try:
            # Get all character references
            all_refs = []
            for char_id in character_ids:
                char_refs = self.get_project_character_references(char_id)
                all_refs.extend(char_refs)
            
            if not (0 <= ref_index < len(all_refs)):
                return "❌ Invalid reference selection", None
            
            # Get single reference image
            ref_image = Image.open(all_refs[ref_index]).convert("RGB")
            
            # Load character data
            characters = []
            for char_id in character_ids:
                character = self.data_manager.load_character(char_id)
                if character:
                    characters.append(character)
            
            if not characters:
                return "❌ No valid characters found", None
            
            # Build prompt for scene variation
            char_descriptions = [f"{char.name}: {char.description}" for char in characters]
            scene_variation_prompt = f"""Create a scene variation based on this reference image.

Scene instruction: {scene_instruction}
Characters: {', '.join(char_descriptions)}

Maintain the same visual style, character identities, and overall composition from the reference while creating the requested scene variation."""
            
            # Generate scene variation
            generated_image, status = self.image_generator.generate_image(scene_variation_prompt, [ref_image])
            
            if generated_image:
                self.temp_generated_image = generated_image
                return f"✅ Scene variation generated from reference", generated_image
            else:
                return f"❌ {status}", None
                
        except Exception as e:
            return f"❌ Error: {str(e)}", None
    
    def edit_image_with_mask(self, base_image: Image.Image, mask_image: Image.Image, instruction: str) -> Tuple[str, Optional[Image.Image]]:
        """Edit image using mask overlay (from c2.py)"""
        try:
            # Build prompt for mask-based editing
            prompt = f"Edit the base image using the mask overlay. White areas in the mask indicate regions to modify.\nEdit instruction: {instruction}\n\nEnsure the edited areas blend naturally with the unmasked portions."
            
            # Convert images to ensure compatibility
            if base_image.mode != "RGB":
                base_image = base_image.convert("RGB")
            if mask_image.mode != "RGB":
                mask_image = mask_image.convert("RGB")
            
            # Generate edited image
            generated_image, status = self.image_generator.generate_image(prompt, [base_image, mask_image])
            
            if generated_image:
                # Store temporarily for potential saving
                self.temp_generated_image = generated_image
                return f"✅ Image edited successfully", generated_image
            else:
                return f"❌ Failed to edit image: {status}", None
        
        except Exception as e:
            return f"❌ Error editing image: {str(e)}", None
    
    def generate_scene(self, character_ids: List[str], scene_instruction: str) -> Tuple[str, Optional[Image.Image]]:
        """Generate multi-character scene"""
        if not character_ids:
            return "❌ Select at least one character", None
        
        if not scene_instruction.strip():
            return "❌ Scene instruction required", None
        
        # Load characters
        characters = []
        reference_images = []
        
        for char_id in character_ids:
            character = self.data_manager.load_character(char_id)
            if character:
                characters.append(character)
                # Use the last reference image for each character
                if character.references:
                    last_ref_path = character.references_dir / character.references[-1]
                    if last_ref_path.exists():
                        try:
                            ref_img = Image.open(last_ref_path)
                            reference_images.append(ref_img)
                        except Exception:
                            pass
        
        if not characters:
            return "❌ No valid characters found", None
        
        # Build scene prompt with style consistency
        character_descriptions = []
        style_notes = []
        
        for char in characters:
            char_desc = f"{char.name}: {char.description}"
            if char.personality:
                char_desc += f" (Personality: {char.personality})"
            character_descriptions.append(char_desc)
            
            # Collect style preferences
            if char.style_notes and char.style_notes.strip():
                style_notes.append(char.style_notes.strip())
        
        # Determine overall scene style
        if style_notes:
            # Use the most common style, or combine if they're compatible
            unique_styles = list(set(style_notes))
            if len(unique_styles) == 1:
                scene_style = unique_styles[0]
            elif any("photorealistic" in style.lower() for style in unique_styles):
                scene_style = "photorealistic 3D"
            elif any("3d" in style.lower() for style in unique_styles):
                scene_style = "3D rendered"
            else:
                scene_style = ", ".join(unique_styles)
        else:
            scene_style = "comic-style"
        
        scene_prompt = f"""Scene: {scene_instruction.strip()}

Characters in scene:
{chr(10).join(character_descriptions)}

Style: {scene_style}

Create a {scene_style} scene showing these characters together in the described scenario. Maintain visual consistency with the specified style throughout the entire scene."""
        
        # Generate scene image
        generated_image, message = self.image_generator.generate_image(scene_prompt, reference_images)
        
        if generated_image and self.current_project:
            # Save scene to project
            scene_path = self.image_generator.save_generated_image(
                generated_image, SCENES_DIR, f"scene_{self.current_project.id}"
            )
            
            scene = Scene(
                id=uuid.uuid4().hex[:8],
                instruction=scene_instruction.strip(),
                characters=character_ids,
                image_path=scene_path.name
            )
            
            self.current_project.scenes.append(scene)
            self.data_manager.save_project(self.current_project)
            
            return f"✅ Scene generated and saved to project: {message}", generated_image
        
        return message, generated_image
    
    def remove_scene(self, scene_index: int) -> str:
        """Remove a scene from the current project"""
        if not self.current_project or not self.current_project.scenes:
            return "❌ No scenes available in current project"
        
        if not (0 <= scene_index < len(self.current_project.scenes)):
            return "❌ Invalid scene index"
        
        # Get scene info before removing
        scene_to_remove = self.current_project.scenes[scene_index]
        scene_name = f"Scene {scene_index + 1}: {scene_to_remove.instruction[:30]}..."
        
        # Remove scene image file if it exists
        if scene_to_remove.image_path:
            image_path = SCENES_DIR / scene_to_remove.image_path
            if image_path.exists():
                try:
                    image_path.unlink()
                except Exception as e:
                    print(f"Warning: Could not delete scene image: {e}")
        
        # Remove scene from project
        self.current_project.scenes.pop(scene_index)
        self.data_manager.save_project(self.current_project)
        
        return f"✅ Removed {scene_name}"
    
    def reorder_scenes(self, new_order: List[int]) -> str:
        """Reorder scenes in the current project"""
        if not self.current_project or not self.current_project.scenes:
            return "❌ No scenes available in current project"
        
        if len(new_order) != len(self.current_project.scenes):
            return "❌ New order must include all scenes"
        
        if not all(0 <= i < len(self.current_project.scenes) for i in new_order):
            return "❌ Invalid scene indices in new order"
        
        if len(set(new_order)) != len(new_order):
            return "❌ Duplicate indices in new order"
        
        # Create new scenes list in the specified order
        old_scenes = self.current_project.scenes.copy()
        self.current_project.scenes = [old_scenes[i] for i in new_order]
        
        self.data_manager.save_project(self.current_project)
        
        return f"✅ Reordered {len(new_order)} scenes"
    
    def get_scene_info(self) -> List[Dict]:
        """Get detailed info about all scenes in current project"""
        if not self.current_project or not self.current_project.scenes:
            return []
        
        scene_info = []
        for i, scene in enumerate(self.current_project.scenes):
            info = {
                "index": i,
                "id": scene.id,
                "instruction": scene.instruction,
                "characters": scene.characters,
                "image_exists": bool(scene.image_path and (SCENES_DIR / scene.image_path).exists()),
                "created_at": scene.created_at
            }
            scene_info.append(info)
        
        return scene_info

    def create_storyboard(self, selected_scene_indices: List[int]) -> Tuple[str, Optional[Image.Image]]:
        """Create storyboard from selected scenes"""
        if not self.current_project or not self.current_project.scenes:
            return "❌ No scenes available in current project", None
        
        if not selected_scene_indices:
            return "❌ Select scenes to include in storyboard", None
        
        # Get selected scenes
        selected_scenes = []
        for idx in selected_scene_indices:
            if 0 <= idx < len(self.current_project.scenes):
                scene = self.current_project.scenes[idx]
                scene_image_path = SCENES_DIR / scene.image_path
                if scene_image_path.exists():
                    selected_scenes.append(scene_image_path)
        
        if not selected_scenes:
            return "❌ No valid scene images found", None
        
        # Create storyboard layout
        try:
            images = [Image.open(path).convert("RGB") for path in selected_scenes]
            
            # Determine layout
            num_panels = len(images)
            if num_panels <= 3:
                cols, rows = num_panels, 1
            elif num_panels <= 6:
                cols, rows = 3, (num_panels + 2) // 3
            else:
                cols, rows = 3, 3
                images = images[:9]  # Limit to 9 panels
            
            # Create storyboard
            panel_width, panel_height = 512, 512
            storyboard_width = cols * panel_width
            storyboard_height = rows * panel_height
            
            storyboard = Image.new("RGB", (storyboard_width, storyboard_height), "white")
            
            for idx, img in enumerate(images):
                row = idx // cols
                col = idx % cols
                
                # Resize image to fit panel
                img_resized = img.resize((panel_width, panel_height), Image.Resampling.LANCZOS)
                
                x = col * panel_width
                y = row * panel_height
                storyboard.paste(img_resized, (x, y))
            
            # Save storyboard
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            storyboard_filename = f"storyboard_{self.current_project.id}_{timestamp}.png"
            storyboard_path = STORYBOARDS_DIR / storyboard_filename
            storyboard.save(storyboard_path)
            
            # Update project
            self.current_project.storyboards.append(storyboard_filename)
            self.data_manager.save_project(self.current_project)
            
            return f"✅ Storyboard created with {len(images)} panels", storyboard
            
        except Exception as e:
            return f"❌ Error creating storyboard: {str(e)}", None
    
    # =============================================================================
    # VOICE & AUDIO METHODS
    # =============================================================================
    
    def configure_elevenlabs_api_key(self, api_key: str) -> str:
        """Configure ElevenLabs API key"""
        self.data_manager.config.elevenlabs_api_key = api_key
        self.data_manager.save_config(self.data_manager.config)
        self.voice_studio = ElevenLabsVoiceStudio(api_key)
        return self.voice_studio.configure_api(api_key)
    
    def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get available ElevenLabs voices"""
        return self.voice_studio.get_available_voices()
    
    def create_character_voice_profile(self, char_id: str, voice_id: str, voice_name: str) -> str:
        """Create voice profile for a character"""
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found"
        
        return self.voice_studio.create_voice_profile(character.name, voice_id, voice_name)
    
    def generate_character_dialogue(self, char_id: str, dialogue_text: str) -> Tuple[str, Optional[str]]:
        """Generate dialogue audio for a character"""
        character = self.data_manager.load_character(char_id)
        if not character:
            return "❌ Character not found", None
        
        success, audio_path, message = self.voice_studio.generate_character_dialogue(character.name, dialogue_text)
        if success and audio_path:
            return message, str(audio_path)
        return message, None
    
    def generate_scene_dialogue_suggestions(self, scene_instruction: str, character_ids: List[str]) -> List[Dict[str, str]]:
        """Generate dialogue suggestions for a scene using AI"""
        if not self.data_manager.config.api_key:
            return []
        
        try:
            # Load characters
            characters = []
            for char_id in character_ids:
                character = self.data_manager.load_character(char_id)
                if character:
                    characters.append(character)
                else:
                    pass  # Character not found
            
            if not characters:
                return []
            
            # Create prompt for dialogue generation
            char_descriptions = "\n".join([
                f"- {char.name}: {char.description[:200]}{'...' if len(char.description) > 200 else ''}"
                for char in characters
            ])
            
            prompt = f"""Generate realistic dialogue for a scene with these characters:

{char_descriptions}

Scene: {scene_instruction}

Create 3-5 lines of dialogue that advance the story and show each character's personality. 
Format as: Character Name: "dialogue text"

Keep dialogue natural and appropriate to each character's voice."""
            
            model = genai.GenerativeModel(TEXT_MODEL)
            response = model.generate_content(prompt)
            
            # Parse dialogue lines
            dialogue_lines = []
            lines = response.text.strip().split('\n')
            
            for line in lines:
                line = line.strip()
                if ':' in line:  # Removed the requirement for quotes to be more flexible
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        character = parts[0].strip().strip('-*•')  # Remove common bullet points
                        text = parts[1].strip().strip('"\'')
                        if character and text:
                            dialogue_lines.append({"character": character, "text": text})
            return dialogue_lines
            
        except Exception as e:
            print(f"Error generating dialogue: {e}")
            return []
    
    def generate_scene_audio(self, scene_instruction: str, dialogue_lines: List[Dict[str, str]]) -> Tuple[str, Optional[Dict[str, Any]]]:
        """Generate complete audio for a scene"""
        success, audio_scene, message = self.voice_studio.generate_scene_audio(scene_instruction, dialogue_lines)
        
        if success and audio_scene:
            scene_data = {
                "id": audio_scene.id,
                "description": audio_scene.scene_description,
                "characters": audio_scene.characters,
                "dialogue_files": [
                    {"character": line["character"], "text": line["text"], "audio_file": line["audio_file"]}
                    for line in audio_scene.dialogue_lines
                ]
            }
            return message, scene_data
        
        return message, None
    
    def generate_storyboard_voiceover(self, storyboard_description: str) -> Tuple[str, Optional[str]]:
        """Generate voiceover narration for storyboard"""
        success, audio_path, message = self.voice_studio.generate_voiceover(storyboard_description)
        if success and audio_path:
            return message, str(audio_path)
        return message, None
    
    def stitch_audio_files(self, audio_files: List[str], output_name: str = "stitched_audio") -> Tuple[str, Optional[str]]:
        """Stitch multiple audio files together"""
        if not audio_files:
            return "❌ No audio files to stitch", None
        
        try:
            # Try to import pydub for audio processing
            try:
                from pydub import AudioSegment
            except ImportError:
                return "❌ Audio stitching requires pydub: pip install pydub", None
            
            # Load all audio files
            combined_audio = AudioSegment.empty()
            valid_files = []
            
            for audio_file in audio_files:
                audio_path = Path(audio_file)
                if audio_path.exists():
                    try:
                        # Add a small pause between audio clips
                        if combined_audio:
                            combined_audio += AudioSegment.silent(duration=500)  # 0.5 second pause
                        
                        # Load and append audio
                        audio_segment = AudioSegment.from_file(str(audio_path))
                        combined_audio += audio_segment
                        valid_files.append(audio_path.name)
                        
                    except Exception as e:
                        print(f"Warning: Could not process audio file {audio_file}: {e}")
                        continue
            
            if not valid_files:
                return "❌ No valid audio files found", None
            
            # Export stitched audio
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{output_name}_{timestamp}.mp3"
            output_path = DIALOGUE_DIR / output_filename
            
            # Export as MP3
            combined_audio.export(str(output_path), format="mp3")
            
            return f"✅ Stitched {len(valid_files)} audio files into {output_filename}", str(output_path)
            
        except Exception as e:
            return f"❌ Audio stitching error: {str(e)}", None
    
    def get_character_voice_profiles(self) -> List[Dict[str, str]]:
        """Get list of character voice profiles"""
        profiles = []
        for profile_id, profile in self.voice_studio.voice_profiles.items():
            profiles.append({
                "id": profile_id,
                "character": profile.character_name,
                "voice": profile.voice_name,
                "created": profile.created_at
            })
        return profiles
    
    def test_voice_connection(self) -> str:
        """Test ElevenLabs API connection"""
        return self.voice_studio.test_api_connection()

# =============================================================================
# GRADIO INTERFACE
# =============================================================================

def create_gradio_interface():
    """Create the comprehensive Gradio interface"""
    
    studio = ComprehensiveStudio()
    
    with gr.Blocks(title="Inspira - Where ideas find their form", theme=gr.themes.Soft()) as app:
        
        gr.Markdown("""
        # 🎨 Inspira - Comprehensive Character & Storyboard Generator Studio
        
        **Features:**
        - 💬 AI-powered character development chat
        - 👥 Persistent character library with reference images
        - 📁 Multi-project management with auto-save
        - 🎵 ElevenLabs voice integration with character dialogue
        - 🎬 Scene audio generation and storyboard voiceover
        - 🎨 Advanced image generation with reference conditioning
        - 🎬 Multi-character scene creation
        - 📖 Storyboard assembly and export
        """)
        
        # Current tab indicator
        current_tab = gr.Textbox(
            value="🔑 API Setup",
            label="Current Tab",
            interactive=False,
            visible=True,
            elem_classes=["current-tab-indicator"]
        )
        
        # =============================================================================
        # API SETUP TAB
        # =============================================================================
        
        with gr.Tab("🔑 API Setup") as api_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Google Gemini API")
                    api_key_input = gr.Textbox(
                        label="Google Gemini API Key",
                        type="password",
                        placeholder="Enter your API key here...",
                        value=studio.data_manager.config.api_key
                    )
                    api_configure_btn = gr.Button("Configure Gemini API", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### ElevenLabs Voice API")
                    elevenlabs_key_input = gr.Textbox(
                        label="ElevenLabs API Key",
                        type="password",
                        placeholder="Enter your ElevenLabs API key...",
                        value=studio.data_manager.config.elevenlabs_api_key
                    )
                    elevenlabs_configure_btn = gr.Button("Configure ElevenLabs API", variant="secondary")
            
            api_status = gr.Markdown()
            
            api_configure_btn.click(
                studio.configure_api_key,
                inputs=[api_key_input],
                outputs=[api_status]
            )
            
            elevenlabs_configure_btn.click(
                studio.configure_elevenlabs_api_key,
                inputs=[elevenlabs_key_input],
                outputs=[api_status]
            )
        
        # =============================================================================
        # PROJECT MANAGER TAB
        # =============================================================================
        
        with gr.Tab("📁 Project Manager") as project_tab:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Create New Project")
                    new_project_name = gr.Textbox(label="Project Name", placeholder="Enter project name...")
                    new_project_desc = gr.Textbox(label="Project Description", placeholder="Optional description...", lines=2)
                    create_project_btn = gr.Button("Create Project", variant="primary")
                
                with gr.Column():
                    gr.Markdown("### Load Existing Project")
                    project_dropdown = gr.Dropdown(
                        label="Select Project",
                        choices=studio.get_project_choices(),
                        value=studio.current_project.id if studio.current_project else None
                    )
                    load_project_btn = gr.Button("Load Project", variant="secondary")
                    refresh_projects_btn = gr.Button("Refresh Projects")
            
            project_status = gr.Markdown()
            current_project_info = gr.Markdown(
                f"**Current Project:** {studio.current_project.name if studio.current_project else 'None'}"
            )
            
            def create_project_handler(name, desc):
                project_id, message = studio.create_project(name, desc)
                choices = studio.get_project_choices()
                current_info = f"**Current Project:** {studio.current_project.name if studio.current_project else 'None'}"
                return message, gr.update(choices=choices, value=project_id), current_info, "", ""
            
            def load_project_handler(project_id):
                message = studio.load_project(project_id)
                current_info = f"**Current Project:** {studio.current_project.name if studio.current_project else 'None'}"
                return message, current_info
            
            def refresh_projects_handler():
                choices = studio.get_project_choices()
                return gr.update(choices=choices)
            
            create_project_btn.click(
                create_project_handler,
                inputs=[new_project_name, new_project_desc],
                outputs=[project_status, project_dropdown, current_project_info, new_project_name, new_project_desc]
            )
            
            load_project_btn.click(
                load_project_handler,
                inputs=[project_dropdown],
                outputs=[project_status, current_project_info]
            )
            
            refresh_projects_btn.click(
                refresh_projects_handler,
                outputs=[project_dropdown]
            )
        
        # =============================================================================
        # CHARACTER DEVELOPMENT TAB
        # =============================================================================
        
        with gr.Tab("💬 Character Development") as char_dev_tab:
            gr.Markdown("### AI-Powered Character Creation Chat")
            gr.Markdown("Describe your character idea and let the AI help you develop it into a complete character profile.")
            
            chatbot = gr.Chatbot(label="Character Development Chat", height=500, type="messages")
            user_input = gr.Textbox(
                label="Your Message",
                placeholder="I want to create a character who is...",
                lines=2
            )
            
            with gr.Row():
                send_chat_btn = gr.Button("Send", variant="primary", scale=1)
                clear_chat_btn = gr.Button("Clear Chat", scale=1)
            
            chat_status = gr.Textbox(label="Status", interactive=False)
            
            with gr.Row():
                save_character_btn = gr.Button("Save Character to Library", variant="primary")
            
            character_name_override = gr.Textbox(
                label="Character Name (override if needed)",
                placeholder="Leave empty to use name from chat..."
            )
            
            def chat_handler(message, history):
                new_history, cleared_input, status = studio.character_developer.develop_character_chat(message, history)
                return new_history, "", status
            
            def save_character_handler(history, name_override):
                message = studio.save_character_from_chat(history, name_override)
                return message, ""
            
            user_input.submit(
                chat_handler,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input, chat_status]
            )
            
            send_chat_btn.click(
                chat_handler,
                inputs=[user_input, chatbot],
                outputs=[chatbot, user_input, chat_status]
            )
            
            clear_chat_btn.click(lambda: ([], ""), outputs=[chatbot, chat_status])
            
            save_character_btn.click(
                save_character_handler,
                inputs=[chatbot, character_name_override],
                outputs=[chat_status, character_name_override]
            )
        
        # =============================================================================
        # CHARACTER LIBRARY TAB
        # =============================================================================
        
        with gr.Tab("👥 Character Library") as char_lib_tab:
            with gr.Row():
                with gr.Column(scale=1):
                    character_dropdown = gr.Dropdown(
                        label="Select Character",
                        choices=studio.get_character_choices()
                    )
                    with gr.Row():
                        refresh_chars_btn = gr.Button("Refresh Characters")
                        new_char_btn = gr.Button("New Character", variant="secondary")
                    
                    gr.Markdown("### Character Details")
                    char_name_input = gr.Textbox(label="Name")
                    char_desc_input = gr.Textbox(label="Physical Description", lines=3)
                    char_personality_input = gr.Textbox(label="Personality", lines=2)
                    char_background_input = gr.Textbox(label="Background", lines=2)
                    char_style_input = gr.Textbox(label="Style Notes", lines=2)
                    
                    save_char_details_btn = gr.Button("Save Character", variant="primary")
                    char_save_status = gr.Textbox(label="Status", interactive=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("### Reference Images")
                    reference_gallery = gr.Gallery(
                        label="Character References", 
                        columns=3,
                        height=300
                    )
                    
                    with gr.Row():
                        upload_reference = gr.Image(label="Upload Reference", type="pil")
                        upload_ref_btn = gr.Button("Add Reference")
            
            def load_character_handler(char_id):
                if not char_id:
                    return "", "", "", "", "", []
                return studio.load_character_details(char_id)
            
            def save_char_handler(char_id, name, desc, personality, background, style):
                # If this is a new character (fields filled but no existing character matches), create new
                if char_id and name.strip():
                    existing_char = studio.data_manager.load_character(char_id)
                    # If the name changed significantly, treat as new character
                    if existing_char and existing_char.name.lower() != name.strip().lower():
                        # Create new character instead of overwriting
                        result = studio.save_character_details("", name, desc, personality, background, style)
                        # Update dropdown
                        return result
                
                return studio.save_character_details(char_id, name, desc, personality, background, style)
            
            def add_reference_handler(char_id, image):
                if not char_id or not image:
                    return "❌ Select character and upload image", []
                return studio.add_reference_image(char_id, image)
            
            def refresh_characters_handler():
                return gr.update(choices=studio.get_character_choices())
            
            character_dropdown.change(
                load_character_handler,
                inputs=[character_dropdown],
                outputs=[char_name_input, char_desc_input, char_personality_input, 
                        char_background_input, char_style_input, reference_gallery]
            )
            
            def save_and_refresh_handler(char_id, name, desc, personality, background, style):
                # Save character
                result = save_char_handler(char_id, name, desc, personality, background, style)
                # Get updated character choices
                new_choices = studio.get_character_choices()
                # Find the ID of the saved character
                saved_char_id = studio.selected_character_id or char_id
                return result, gr.update(choices=new_choices, value=saved_char_id)
            
            save_char_details_btn.click(
                save_and_refresh_handler,
                inputs=[character_dropdown, char_name_input, char_desc_input, 
                       char_personality_input, char_background_input, char_style_input],
                outputs=[char_save_status, character_dropdown]
            )
            
            upload_ref_btn.click(
                add_reference_handler,
                inputs=[character_dropdown, upload_reference],
                outputs=[char_save_status, reference_gallery]
            )
            
            def new_character_handler():
                # Clear all fields and dropdown selection
                return (gr.update(value=None), "", "", "", "", "", [])
            
            refresh_chars_btn.click(
                refresh_characters_handler,
                outputs=[character_dropdown]
            )
            
            new_char_btn.click(
                new_character_handler,
                outputs=[character_dropdown, char_name_input, char_desc_input, 
                        char_personality_input, char_background_input, char_style_input, reference_gallery]
            )
        
        # =============================================================================
        # IMAGE GENERATION TAB
        # =============================================================================
        
        with gr.Tab("🎨 Image Generation") as img_gen_tab:
            gr.Markdown("### Project-Based Character Image Generation")
            gr.Markdown("💡 **Tip**: Upload reference images and save generated images to build a character reference library for consistent results!")
            
            with gr.Row():
                with gr.Column():
                    gen_character_select = gr.Dropdown(
                        label="Select Character",
                        choices=studio.get_character_choices()
                    )
                    refresh_gen_chars = gr.Button("Refresh Characters")
                    
                    prompt_modifier = gr.Textbox(
                        label="Generation Prompt",
                        placeholder="3/4 view, dramatic lighting, action pose...",
                        lines=3
                    )
                    
                    gr.Markdown("#### Reference Images")
                    reference_upload = gr.Image(
                        label="Upload Reference Image (Optional)",
                        type="pil"
                    )
                    
                    with gr.Row():
                        upload_ref_btn = gr.Button("Add to References", size="sm")
                        use_references_check = gr.Checkbox(
                            label="Use Existing References",
                            value=True
                        )
                    
                    
                    with gr.Row():
                        generate_image_btn = gr.Button("Generate Image", variant="primary")
                        save_to_refs_btn = gr.Button("Save Generated → References", variant="secondary")
                
                with gr.Column():
                    generated_image_output = gr.Image(label="Generated Image")
                    
                    # Save/Discard options for generated image
                    with gr.Row():
                        save_generated_btn = gr.Button("Save to References", variant="primary", visible=False)
                        discard_generated_btn = gr.Button("Discard", variant="secondary", visible=False)
                    
                    save_description = gr.Textbox(
                        label="Description for Reference Library (Optional)",
                        placeholder="e.g., 'front view, happy expression'",
                        lines=2,
                        visible=False
                    )
                    
                    generation_status = gr.Textbox(label="Status", interactive=False)
                    
                    # Saved Reference Images Section
                    gr.Markdown("#### Saved Reference Images")
                    saved_ref_gallery = gr.Gallery(
                        label="Select up to 3 references for generation (click to select/deselect)",
                        columns=3,
                        height=200,
                        allow_preview=True,
                        interactive=True
                    )
                    
                    selected_refs_info = gr.Markdown("**Selected References:** None")
                    
                    with gr.Row():
                        clear_ref_selection_btn = gr.Button("Clear Selection", size="sm")
                        delete_selected_refs_btn = gr.Button("Delete Selected", size="sm", variant="stop")
                        single_ref_mode = gr.Checkbox(
                            label="Single Reference Mode (for iterative changes)",
                            value=False
                        )
            
            # Store selected reference indices
            selected_ref_indices = []
            
            def generate_image_handler(char_id, prompt, use_refs, single_mode):
                if not char_id:
                    return "❌ Select a character first", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                # Use selected references or all references based on mode
                if selected_ref_indices and single_mode and len(selected_ref_indices) == 1:
                    # Single reference mode for iterative changes
                    status, image = studio.generate_with_single_reference(char_id, prompt, selected_ref_indices[0])
                elif selected_ref_indices:
                    # Use selected references (up to 3)
                    status, image = studio.generate_with_selected_references(char_id, prompt, selected_ref_indices[:3])
                else:
                    # Fallback to existing method
                    status, image = studio.generate_character_image(char_id, prompt, use_refs)
                
                if image:
                    # Show save/discard buttons after successful generation
                    return status, image, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                else:
                    return status, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            def save_generated_handler(char_id, description):
                if not char_id:
                    return "❌ Select a character first", [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                result, gallery_paths = studio.save_generated_to_references(char_id, description)
                
                # Hide save/discard buttons after saving
                return result, gallery_paths, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            def discard_generated_handler():
                # Clear the generated image and hide buttons
                studio.temp_generated_image = None
                return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            def handle_gallery_selection(evt: gr.SelectData):
                nonlocal selected_ref_indices
                
                if evt.index in selected_ref_indices:
                    # Deselect if already selected
                    selected_ref_indices.remove(evt.index)
                else:
                    # Select if not selected (max 3)
                    if len(selected_ref_indices) < 3:
                        selected_ref_indices.append(evt.index)
                
                # Update status
                if not selected_ref_indices:
                    status = "**Selected References:** None"
                else:
                    count = len(selected_ref_indices)
                    status = f"**Selected References:** {count}/3 selected"
                
                return status
            
            def clear_selection_handler():
                nonlocal selected_ref_indices
                selected_ref_indices = []
                return "**Selected References:** None"
            
            def delete_selected_refs_handler(char_id):
                nonlocal selected_ref_indices
                if not char_id:
                    return "❌ Please select a character first", [], "**Selected References:** None"
                
                if not selected_ref_indices:
                    return "❌ Please select references to delete", [], "**Selected References:** None"
                
                result, gallery_paths = studio.delete_reference_from_project_character(char_id, selected_ref_indices.copy())
                selected_ref_indices = []  # Clear selection after deletion
                
                return result, gallery_paths, "**Selected References:** None"
            
            def save_to_refs_handler(char_id, description):
                if not char_id:
                    return "❌ Select a character first", []
                result, gallery_paths = studio.save_generated_to_references(char_id, description)
                return result, gr.update(value=gallery_paths), ""
            
            def upload_reference_handler(char_id, image, description):
                if not char_id:
                    return "❌ Select a character first", []
                if not image:
                    return "❌ Please upload an image", []
                
                result, gallery_paths = studio.add_reference_to_project_character(char_id, image, description)
                return result, gr.update(value=gallery_paths)
            
            def load_character_references(char_id):
                if not char_id:
                    return []
                return gr.update(value=studio.get_project_character_references(char_id))
            
            def refresh_gen_chars_handler():
                choices = studio.get_character_choices()
                return gr.update(choices=choices)
            
            # Event bindings
            generate_image_btn.click(
                generate_image_handler,
                inputs=[gen_character_select, prompt_modifier, use_references_check, single_ref_mode],
                outputs=[generation_status, generated_image_output, save_generated_btn, discard_generated_btn, save_description]
            )
            
            save_generated_btn.click(
                save_generated_handler,
                inputs=[gen_character_select, save_description],
                outputs=[generation_status, saved_ref_gallery, save_generated_btn, discard_generated_btn, save_description]
            )
            
            discard_generated_btn.click(
                discard_generated_handler,
                outputs=[generated_image_output, save_generated_btn, discard_generated_btn, save_description]
            )
            
            save_to_refs_btn.click(
                save_to_refs_handler,
                inputs=[gen_character_select, save_description],
                outputs=[generation_status, saved_ref_gallery]
            )
            
            upload_ref_btn.click(
                upload_reference_handler,
                inputs=[gen_character_select, reference_upload, save_description],
                outputs=[generation_status, saved_ref_gallery]
            )
            
            saved_ref_gallery.select(
                handle_gallery_selection,
                outputs=[selected_refs_info]
            )
            
            clear_ref_selection_btn.click(
                clear_selection_handler,
                outputs=[selected_refs_info]
            )
            
            delete_selected_refs_btn.click(
                delete_selected_refs_handler,
                inputs=[gen_character_select],
                outputs=[generation_status, saved_ref_gallery, selected_refs_info]
            )
            
            gen_character_select.change(
                load_character_references,
                inputs=[gen_character_select],
                outputs=[saved_ref_gallery]
            )
            
            refresh_gen_chars.click(
                refresh_gen_chars_handler,
                outputs=[gen_character_select]
            )
        
        # =============================================================================
        # CHAT-BASED IMAGE GENERATION TAB
        # =============================================================================
        
        with gr.Tab("💬 Chat Image Generation") as chat_img_tab:
            gr.Markdown("### Iterative Chat-Based Character Image Generation")
            gr.Markdown("🗣️ **New Feature**: Generate images through conversation! Chat with the AI to iteratively refine your character images. Each generation builds on your conversation history.")
            
            with gr.Row():
                with gr.Column():
                    chat_character_select = gr.Dropdown(
                        label="Select Character for Chat",
                        choices=studio.get_character_choices()
                    )
                    refresh_chat_chars = gr.Button("Refresh Characters")
                    
                    # Chat interface
                    chat_history = gr.Chatbot(
                        label="Character Image Generation Chat",
                        height=400,
                        type="messages"
                    )
                    
                    chat_input = gr.Textbox(
                        label="Describe what you want to generate or modify",
                        placeholder="Generate a portrait of this character... or Make them look angry... or Change the lighting to dramatic...",
                        lines=2
                    )
                    
                    with gr.Row():
                        chat_generate_btn = gr.Button("Generate Image", variant="primary")
                        clear_chat_btn = gr.Button("Clear Chat History", variant="secondary")
                
                with gr.Column():
                    chat_generated_image = gr.Image(label="Generated Image")
                    
                    # Save/Discard options for chat generated image
                    with gr.Row():
                        save_chat_generated_btn = gr.Button("Save to References", variant="primary", visible=False)
                        discard_chat_generated_btn = gr.Button("Discard", variant="secondary", visible=False)
                    
                    save_chat_description = gr.Textbox(
                        label="Reference Description (Optional)",
                        placeholder="Portrait, angry expression, dramatic lighting...",
                        lines=1,
                        visible=False
                    )
                    
                    chat_status = gr.Markdown("Select a character and start chatting to generate images!")
                    
                    # Saved Reference Images Section for Chat
                    gr.Markdown("#### Saved Reference Images")
                    chat_saved_ref_gallery = gr.Gallery(
                        label="Select up to 3 references for generation (click to select/deselect)",
                        columns=3,
                        height=200,
                        allow_preview=True,
                        interactive=True
                    )
                    
                    chat_selected_refs_info = gr.Markdown("**Selected References:** None")
                    
                    with gr.Row():
                        clear_chat_ref_selection_btn = gr.Button("Clear Selection", size="sm")
                        delete_chat_selected_refs_btn = gr.Button("Delete Selected", size="sm", variant="stop")
                        chat_single_ref_mode = gr.Checkbox(
                            label="Single Reference Mode (for 'keep same but add...' prompts)",
                            value=False
                        )
            
            # Store selected reference indices for chat
            chat_selected_ref_indices = []
            
            def chat_generate_handler(char_id: str, instruction: str, history: List, single_mode: bool):
                if not char_id:
                    return history, "❌ Please select a character first", None, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                # Use selected references for generation
                if chat_selected_ref_indices and single_mode and len(chat_selected_ref_indices) == 1:
                    # Single reference mode for iterative changes
                    status, generated_image = studio.generate_with_single_reference(char_id, instruction, chat_selected_ref_indices[0])
                elif chat_selected_ref_indices:
                    # Use selected references (up to 3)
                    status, generated_image = studio.generate_with_selected_references(char_id, instruction, chat_selected_ref_indices[:3])
                else:
                    # Fallback to existing chat method
                    status, generated_image, updated_history = studio.chat_generate_image(char_id, instruction)
                    
                    # Convert chat history to messages format for display
                    display_history = []
                    for msg_tuple in updated_history:
                        role, content = msg_tuple
                        if isinstance(content, list):
                            content = content[0] if content else ""
                        display_history.append({"role": role, "content": content})
                    
                    if generated_image:
                        return display_history, status, generated_image, "", gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                    else:
                        return display_history, status, None, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                # For reference-based generation, update chat history manually
                character = studio.data_manager.load_character(char_id)
                if character:
                    character.chat_history.append(("user", instruction))
                    character.chat_history.append(("assistant", status))
                    studio.data_manager.save_character(character)
                    
                    # Convert to display format
                    display_history = []
                    for msg_tuple in character.chat_history:
                        role, content = msg_tuple
                        if isinstance(content, list):
                            content = content[0] if content else ""
                        display_history.append({"role": role, "content": content})
                else:
                    display_history = history
                
                if generated_image:
                    return display_history, status, generated_image, "", gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                else:
                    return display_history, status, None, "", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            def clear_chat_handler(char_id: str):
                if not char_id:
                    return [], "❌ Please select a character first"
                
                character = studio.data_manager.load_character(char_id)
                if character:
                    character.chat_history = []
                    studio.data_manager.save_character(character)
                    return [], "✅ Chat history cleared"
                return [], "❌ Character not found"
            
            def load_chat_history(char_id: str):
                if not char_id:
                    return [], []
                
                character = studio.data_manager.load_character(char_id)
                if not character:
                    return [], []
                
                # Convert chat history to messages format
                display_history = []
                for msg_tuple in character.chat_history:
                    role, content = msg_tuple
                    if isinstance(content, list):  # Handle nested lists from JSON
                        content = content[0] if content else ""
                    display_history.append({"role": role, "content": content})
                
                # Load references
                references = studio.get_project_character_references(char_id)
                
                return display_history, references
            
            def save_chat_generated_handler(char_id: str, description: str):
                if not char_id:
                    return "❌ Please select a character first", [], gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                result, gallery_paths = studio.save_generated_to_references(char_id, description)
                return result, gallery_paths, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            def discard_chat_generated_handler():
                # Clear the generated image and hide buttons
                studio.temp_generated_image = None
                return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            def handle_chat_gallery_selection(evt: gr.SelectData):
                nonlocal chat_selected_ref_indices
                
                if evt.index in chat_selected_ref_indices:
                    # Deselect if already selected
                    chat_selected_ref_indices.remove(evt.index)
                else:
                    # Select if not selected (max 3)
                    if len(chat_selected_ref_indices) < 3:
                        chat_selected_ref_indices.append(evt.index)
                
                # Update status
                if not chat_selected_ref_indices:
                    status = "**Selected References:** None"
                else:
                    count = len(chat_selected_ref_indices)
                    status = f"**Selected References:** {count}/3 selected"
                
                return status
            
            def clear_chat_selection_handler():
                nonlocal chat_selected_ref_indices
                chat_selected_ref_indices = []
                return "**Selected References:** None"
            
            def delete_chat_selected_refs_handler(char_id):
                nonlocal chat_selected_ref_indices
                if not char_id:
                    return "❌ Please select a character first", [], "**Selected References:** None"
                
                if not chat_selected_ref_indices:
                    return "❌ Please select references to delete", [], "**Selected References:** None"
                
                result, gallery_paths = studio.delete_reference_from_project_character(char_id, chat_selected_ref_indices.copy())
                chat_selected_ref_indices = []  # Clear selection after deletion
                
                return result, gallery_paths, "**Selected References:** None"
            
            def save_chat_to_refs_handler(char_id: str, description: str):
                if not char_id:
                    return "❌ Please select a character first", []
                
                result, gallery_paths = studio.save_generated_to_references(char_id, description)
                return result, gallery_paths
            
            # Bind events
            chat_generate_btn.click(
                chat_generate_handler,
                inputs=[chat_character_select, chat_input, chat_history, chat_single_ref_mode],
                outputs=[chat_history, chat_status, chat_generated_image, chat_input, save_chat_generated_btn, discard_chat_generated_btn, save_chat_description]
            )
            
            chat_input.submit(
                chat_generate_handler,
                inputs=[chat_character_select, chat_input, chat_history, chat_single_ref_mode],
                outputs=[chat_history, chat_status, chat_generated_image, chat_input, save_chat_generated_btn, discard_chat_generated_btn, save_chat_description]
            )
            
            save_chat_generated_btn.click(
                save_chat_generated_handler,
                inputs=[chat_character_select, save_chat_description],
                outputs=[chat_status, chat_saved_ref_gallery, save_chat_generated_btn, discard_chat_generated_btn, save_chat_description]
            )
            
            discard_chat_generated_btn.click(
                discard_chat_generated_handler,
                outputs=[chat_generated_image, save_chat_generated_btn, discard_chat_generated_btn, save_chat_description]
            )
            
            chat_saved_ref_gallery.select(
                handle_chat_gallery_selection,
                outputs=[chat_selected_refs_info]
            )
            
            clear_chat_ref_selection_btn.click(
                clear_chat_selection_handler,
                outputs=[chat_selected_refs_info]
            )
            
            delete_chat_selected_refs_btn.click(
                delete_chat_selected_refs_handler,
                inputs=[chat_character_select],
                outputs=[chat_status, chat_saved_ref_gallery, chat_selected_refs_info]
            )
            
            clear_chat_btn.click(
                clear_chat_handler,
                inputs=[chat_character_select],
                outputs=[chat_history, chat_status]
            )
            
            chat_character_select.change(
                load_chat_history,
                inputs=[chat_character_select],
                outputs=[chat_history, chat_saved_ref_gallery]
            )
            
            refresh_chat_chars.click(
                lambda: gr.update(choices=studio.get_character_choices()),
                outputs=[chat_character_select]
            )
        
        # =============================================================================
        # SCENE CREATOR TAB
        # =============================================================================
        
        with gr.Tab("🎬 Scene Creator") as scene_tab:
            gr.Markdown("### Multi-Character Scene Generation with Reference Control")
            gr.Markdown("🎨 **Enhanced**: Select characters, choose their reference images, and create scenes with full visual control.")
            
            with gr.Row():
                with gr.Column():
                    scene_characters_select = gr.CheckboxGroup(
                        label="Select Characters for Scene",
                        choices=[name for name, _ in studio.get_character_choices()]
                    )
                    refresh_scene_chars = gr.Button("Refresh Characters")
                    
                    scene_instruction = gr.Textbox(
                        label="Scene Description",
                        placeholder="Describe what should happen in this scene...",
                        lines=4
                    )
                    
                    # Reference Images Section for Selected Characters
                    gr.Markdown("#### Character Reference Images")
                    scene_character_refs_display = gr.HTML("Select characters to see their reference images")
                    
                    scene_char_ref_galleries = {}  # Will be populated dynamically
                    
                    with gr.Row():
                        generate_scene_btn = gr.Button("Generate Scene", variant="primary")
                        use_selected_refs_only = gr.Checkbox(
                            label="Use only selected references",
                            value=False
                        )
                
                with gr.Column():
                    scene_image_output = gr.Image(label="Generated Scene")
                    
                    # Save/Discard options for scene
                    with gr.Row():
                        save_scene_btn = gr.Button("Save Scene", variant="primary", visible=False)
                        add_scene_to_ref_btn = gr.Button("Add to References", variant="secondary", visible=False)
                        discard_scene_btn = gr.Button("Discard", variant="secondary", visible=False)
                    
                    scene_save_description = gr.Textbox(
                        label="Scene Description (Optional)",
                        placeholder="Two characters talking in a park...",
                        lines=1,
                        visible=False
                    )
                    
                    # Character selector for adding to references
                    scene_ref_char_select = gr.Dropdown(
                        label="Add to Character References",
                        choices=studio.get_character_choices(),
                        visible=False
                    )
                    
                    scene_status = gr.Textbox(label="Scene Status", interactive=False)
                    
                    # Reference Images for All Selected Characters
                    gr.Markdown("#### Selected Characters' References")
                    scene_all_refs_gallery = gr.Gallery(
                        label="All references from selected characters (click to select/deselect for scene generation)",
                        columns=4,
                        height=300,
                        allow_preview=True,
                        interactive=True
                    )
                    
                    scene_selected_refs_info = gr.Markdown("**Selected References:** None")
                    
                    with gr.Row():
                        clear_scene_refs_btn = gr.Button("Clear Reference Selection", size="sm")
                        scene_single_ref_mode = gr.Checkbox(
                            label="Single Reference Mode (for scene variations)",
                            value=False
                        )
            
            # Store selected reference indices for scene generation
            scene_selected_ref_indices = []
            scene_char_to_refs_map = {}  # Maps character IDs to their reference paths
            
            def load_character_references_for_scene(selected_char_names):
                """Load and display references for selected characters"""
                if not selected_char_names:
                    return [], "Select characters to see their reference images", "**Selected References:** None"
                
                # Get character IDs
                char_choices = dict(studio.get_character_choices())
                character_ids = []
                for name in selected_char_names:
                    for choice_name, choice_id in char_choices.items():
                        if choice_name == name:
                            character_ids.append(choice_id)
                            break
                
                # Collect all references from selected characters
                all_refs = []
                scene_char_to_refs_map.clear()
                
                char_info_html = "<div style='font-size: 14px;'>"
                
                for char_id in character_ids:
                    char_refs = studio.get_project_character_references(char_id)
                    if char_refs:
                        scene_char_to_refs_map[char_id] = char_refs
                        all_refs.extend(char_refs)
                        
                        # Add character info
                        character = studio.data_manager.load_character(char_id)
                        char_name = character.name if character else "Unknown"
                        char_info_html += f"<b>{char_name}:</b> {len(char_refs)} references<br>"
                    else:
                        character = studio.data_manager.load_character(char_id)
                        char_name = character.name if character else "Unknown"
                        char_info_html += f"<b>{char_name}:</b> No references available<br>"
                
                char_info_html += "</div>"
                
                return all_refs, char_info_html, "**Selected References:** None"
            
            def generate_scene_handler(selected_char_names, instruction, use_selected_only, single_mode):
                # Convert character names to IDs
                char_choices = dict(studio.get_character_choices())
                char_ids = []
                for name in selected_char_names:
                    for choice_name, choice_id in char_choices.items():
                        if choice_name == name:
                            char_ids.append(choice_id)
                            break
                
                if not char_ids:
                    return "❌ Select at least one character", None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                # Enhanced scene generation with reference control
                if scene_selected_ref_indices and use_selected_only:
                    if single_mode and len(scene_selected_ref_indices) == 1:
                        # Single reference mode for scene variations
                        status, image = studio.generate_scene_with_single_reference(char_ids, instruction, scene_selected_ref_indices[0])
                    else:
                        # Use selected references only
                        status, image = studio.generate_scene_with_selected_references(char_ids, instruction, scene_selected_ref_indices)
                else:
                    # Fallback to existing method (uses automatic reference selection)
                    status, image = studio.generate_scene(char_ids, instruction)
                
                if image:
                    return status, image, gr.update(visible=True), gr.update(visible=True), gr.update(visible=True), gr.update(visible=True)
                else:
                    return status, None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            def handle_scene_gallery_selection(evt: gr.SelectData):
                nonlocal scene_selected_ref_indices
                
                if evt.index in scene_selected_ref_indices:
                    # Deselect if already selected
                    scene_selected_ref_indices.remove(evt.index)
                else:
                    # Select if not selected
                    scene_selected_ref_indices.append(evt.index)
                
                # Update status
                if not scene_selected_ref_indices:
                    status = "**Selected References:** None"
                else:
                    count = len(scene_selected_ref_indices)
                    status = f"**Selected References:** {count} selected"
                
                return status
            
            def clear_scene_selection_handler():
                nonlocal scene_selected_ref_indices
                scene_selected_ref_indices = []
                return "**Selected References:** None"
            
            def save_scene_handler(description):
                if not studio.temp_generated_image:
                    return "❌ No scene to save", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
                
                # Save to scenes directory with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"scene_{timestamp}_{uuid.uuid4().hex[:6]}.png"
                scene_path = SCENES_DIR / filename
                
                studio.temp_generated_image.save(scene_path)
                studio.temp_generated_image = None
                
                return f"✅ Scene saved as {filename}", gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            def add_scene_to_references_handler(char_id):
                if not studio.temp_generated_image:
                    return "❌ No scene to save to references"
                
                if not char_id:
                    return "❌ Please select a character"
                
                result, _ = studio.add_reference_to_project_character(
                    char_id, studio.temp_generated_image, "Generated scene"
                )
                return result
            
            def discard_scene_handler():
                # Clear the generated scene and hide buttons
                studio.temp_generated_image = None
                return None, gr.update(visible=False), gr.update(visible=False), gr.update(visible=False), gr.update(visible=False)
            
            # Event bindings for enhanced scene creator
            generate_scene_btn.click(
                generate_scene_handler,
                inputs=[scene_characters_select, scene_instruction, use_selected_refs_only, scene_single_ref_mode],
                outputs=[scene_status, scene_image_output, save_scene_btn, add_scene_to_ref_btn, discard_scene_btn, scene_ref_char_select]
            )
            
            save_scene_btn.click(
                save_scene_handler,
                inputs=[scene_save_description],
                outputs=[scene_status, save_scene_btn, add_scene_to_ref_btn, discard_scene_btn, scene_ref_char_select]
            )
            
            discard_scene_btn.click(
                discard_scene_handler,
                outputs=[scene_image_output, save_scene_btn, add_scene_to_ref_btn, discard_scene_btn, scene_ref_char_select]
            )
            
            add_scene_to_ref_btn.click(
                add_scene_to_references_handler,
                inputs=[scene_ref_char_select],
                outputs=[scene_status]
            )
            
            scene_characters_select.change(
                load_character_references_for_scene,
                inputs=[scene_characters_select],
                outputs=[scene_all_refs_gallery, scene_character_refs_display, scene_selected_refs_info]
            )
            
            scene_all_refs_gallery.select(
                handle_scene_gallery_selection,
                outputs=[scene_selected_refs_info]
            )
            
            clear_scene_refs_btn.click(
                clear_scene_selection_handler,
                outputs=[scene_selected_refs_info]
            )
            
            refresh_scene_chars.click(
                lambda: gr.update(choices=[name for name, _ in studio.get_character_choices()]),
                outputs=[scene_characters_select]
            )
        
        # =============================================================================
        # MASK-BASED IMAGE EDITING TAB
        # =============================================================================
        
        with gr.Tab("✏️ Mask Editor") as mask_tab:
            gr.Markdown("### Mask-Based Image Editing")
            gr.Markdown("🎨 **Advanced Feature**: Upload a base image and a mask to selectively edit portions of the image. White areas in the mask will be modified, black areas will be preserved.")
            
            with gr.Row():
                with gr.Column():
                    base_image_input = gr.Image(
                        label="Base Image (to be edited)",
                        type="pil"
                    )
                    
                    mask_image_input = gr.Image(
                        label="Mask Image (white=edit, black=preserve)",
                        type="pil"
                    )
                    
                    edit_instruction = gr.Textbox(
                        label="Edit Instruction",
                        placeholder="Change the background to a sunset scene... or Make the character's expression happy... or Add dramatic lighting...",
                        lines=3
                    )
                    
                    edit_image_btn = gr.Button("Apply Mask Edit", variant="primary")
                
                with gr.Column():
                    edited_image_output = gr.Image(label="Edited Result")
                    
                    with gr.Row():
                        save_edited_btn = gr.Button("Save to Character References", variant="secondary")
                        edited_char_select = gr.Dropdown(
                            label="Character to Save To",
                            choices=studio.get_character_choices()
                        )
                    
                    edit_description = gr.Textbox(
                        label="Description for Reference (Optional)",
                        placeholder="Edited with dramatic lighting...",
                        lines=1
                    )
                    
                    edit_status = gr.Markdown("Upload base image and mask, then describe your edit!")
            
            # Mask editing handlers
            def mask_edit_handler(base_img: Image.Image, mask_img: Image.Image, instruction: str):
                if not base_img:
                    return "❌ Please upload a base image", None
                if not mask_img:
                    return "❌ Please upload a mask image", None
                if not instruction.strip():
                    return "❌ Please provide edit instruction", None
                
                status, edited_image = studio.edit_image_with_mask(base_img, mask_img, instruction)
                return status, edited_image
            
            def save_edited_handler(char_id: str, description: str):
                if not char_id:
                    return "❌ Please select a character to save to"
                
                if not studio.temp_generated_image:
                    return "❌ No edited image to save"
                
                result, _ = studio.add_reference_to_project_character(
                    char_id, studio.temp_generated_image, description or "Mask-edited image"
                )
                return result
            
            # Bind mask editing events
            edit_image_btn.click(
                mask_edit_handler,
                inputs=[base_image_input, mask_image_input, edit_instruction],
                outputs=[edit_status, edited_image_output]
            )
            
            save_edited_btn.click(
                save_edited_handler,
                inputs=[edited_char_select, edit_description],
                outputs=[edit_status]
            )
        
        # =============================================================================
        # IMAGE EXPANSION TAB
        # =============================================================================
        
        with gr.Tab("📐 Image Expansion") as expansion_tab:
            gr.Markdown("### Image Expansion/Outpainting for Camera Angles")
            gr.Markdown("🔍 **Camera Angles**: Expand images in any direction to show more of the scene, perfect for creating different camera perspectives from the same base image.")
            
            with gr.Row():
                with gr.Column():
                    expand_base_image = gr.Image(
                        label="Base Image to Expand",
                        type="pil"
                    )
                    
                    with gr.Row():
                        expand_direction = gr.Radio(
                            choices=["left", "right", "up", "down"],
                            label="Expansion Direction",
                            value="right"
                        )
                        expansion_amount = gr.Textbox(
                            label="Expansion Amount",
                            placeholder="e.g., '200' (pixels) or '30%' (percentage)",
                            value="25%"
                        )
                    
                    expand_instruction = gr.Textbox(
                        label="Expansion Instruction (Optional)",
                        placeholder="e.g., 'Show more of the background environment' or 'Reveal what's to the left of the character'",
                        lines=3
                    )
                    
                    expand_image_btn = gr.Button("Expand Image", variant="primary")
                
                with gr.Column():
                    expanded_image_output = gr.Image(label="Expanded Result")
                    
                    with gr.Row():
                        save_expanded_btn = gr.Button("Save to References", variant="secondary")
                        expand_char_select = gr.Dropdown(
                            label="Character to Save To",
                            choices=studio.get_character_choices()
                        )
                    
                    expand_description = gr.Textbox(
                        label="Description for Reference (Optional)",
                        placeholder="Wider angle view, expanded right...",
                        lines=1
                    )
                    
                    expand_status = gr.Markdown("Upload an image and choose expansion settings!")
                    
                    # Quick expansion presets
                    gr.Markdown("#### Quick Presets")
                    with gr.Row():
                        preset_wide_left = gr.Button("Widen Left 25%", size="sm")
                        preset_wide_right = gr.Button("Widen Right 25%", size="sm")
                        preset_taller_up = gr.Button("Taller Up 30%", size="sm")
                        preset_taller_down = gr.Button("Taller Down 30%", size="sm")
            
            # Expansion handlers
            def expand_image_handler(base_img: Image.Image, direction: str, amount: str, instruction: str):
                if not base_img:
                    return "❌ Please upload a base image", None
                if not direction:
                    return "❌ Please select expansion direction", None
                if not amount.strip():
                    return "❌ Please enter expansion amount", None
                
                status, expanded_image = studio.expand_image(base_img, direction, amount.strip(), instruction)
                return status, expanded_image
            
            def save_expanded_handler(char_id: str, description: str):
                if not char_id:
                    return "❌ Please select a character to save to"
                
                if not studio.temp_generated_image:
                    return "❌ No expanded image to save"
                
                result, _ = studio.add_reference_to_project_character(
                    char_id, studio.temp_generated_image, description or "Expanded image"
                )
                return result
            
            def preset_expand_handler(direction: str, amount: str):
                return direction, amount
            
            # Bind expansion events
            expand_image_btn.click(
                expand_image_handler,
                inputs=[expand_base_image, expand_direction, expansion_amount, expand_instruction],
                outputs=[expand_status, expanded_image_output]
            )
            
            save_expanded_btn.click(
                save_expanded_handler,
                inputs=[expand_char_select, expand_description],
                outputs=[expand_status]
            )
            
            # Preset buttons
            preset_wide_left.click(
                lambda: preset_expand_handler("left", "25%"),
                outputs=[expand_direction, expansion_amount]
            )
            
            preset_wide_right.click(
                lambda: preset_expand_handler("right", "25%"),
                outputs=[expand_direction, expansion_amount]
            )
            
            preset_taller_up.click(
                lambda: preset_expand_handler("up", "30%"),
                outputs=[expand_direction, expansion_amount]
            )
            
            preset_taller_down.click(
                lambda: preset_expand_handler("down", "30%"),
                outputs=[expand_direction, expansion_amount]
            )
        
        # =============================================================================
        # STORYBOARD BUILDER TAB
        # =============================================================================
        
        with gr.Tab("📖 Storyboard Builder") as storyboard_tab:
            gr.Markdown("### Scene Management & Storyboard Creation")
            
            def get_current_scenes():
                if studio.current_project and studio.current_project.scenes:
                    return [f"Scene {i+1}: {scene.instruction[:50]}..." 
                           for i, scene in enumerate(studio.current_project.scenes)]
                return []
            
            def get_scene_details():
                """Get detailed scene information for management"""
                if not studio.current_project:
                    return "No project loaded"
                
                scene_info = studio.get_scene_info()
                if not scene_info:
                    return "No scenes in current project"
                
                details = []
                for info in scene_info:
                    status = "✅" if info["image_exists"] else "❌"
                    char_names = ", ".join(info["characters"][:2]) + ("..." if len(info["characters"]) > 2 else "")
                    details.append(f"{status} Scene {info['index']+1}: {info['instruction'][:40]}... (Characters: {char_names})")
                
                return "\n".join(details)
            
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Scene Management")
                    scene_details = gr.Textbox(
                        label="Current Scenes",
                        value=get_scene_details(),
                        lines=6,
                        interactive=False
                    )
                    
                    with gr.Row():
                        refresh_scenes_btn = gr.Button("Refresh Scenes")
                        remove_scene_btn = gr.Button("Remove Scene", variant="stop")
                    
                    scene_to_remove = gr.Number(
                        label="Scene Number to Remove (1-based)",
                        step=1,
                        precision=0,
                        value=None
                    )
                    
                    gr.Markdown("#### Reorder Scenes")
                    reorder_input = gr.Textbox(
                        label="New Scene Order (comma-separated, 1-based)",
                        placeholder="e.g., 3,1,2,4",
                        info="Enter the new order as numbers separated by commas"
                    )
                    reorder_btn = gr.Button("Reorder Scenes")
                    
                with gr.Column():
                    gr.Markdown("#### Create Storyboard")
                    scenes_checklist = gr.CheckboxGroup(
                        label="Select Scenes for Storyboard",
                        choices=get_current_scenes()
                    )
                    
                    create_storyboard_btn = gr.Button("Create Storyboard", variant="primary")
                    
                    storyboard_output = gr.Image(label="Generated Storyboard")
                    
                    # Audio integration for storyboard
                    gr.Markdown("#### Storyboard Audio")
                    with gr.Row():
                        auto_generate_dialogue_btn = gr.Button("🎭 Auto-Generate Scene Dialogue", variant="secondary")
                        generate_storyboard_audio_btn = gr.Button("🔊 Create Audio Storyboard", variant="secondary")
                    
                    storyboard_audio_status = gr.Textbox(label="Audio Generation Status", interactive=False)
                    
                    # Dialogue display
                    storyboard_dialogue_display = gr.Textbox(
                        label="Generated Dialogue",
                        placeholder="Generated dialogue will appear here...",
                        lines=10,
                        interactive=False,
                        visible=False
                    )
                    
                    # Stitched audio playback
                    with gr.Group():
                        gr.Markdown("**Complete Storyboard Audio:**")
                        storyboard_stitched_player = gr.Audio(label="🎧 Play Complete Storyboard", visible=False)
                    
                    generated_storyboard_audio = gr.File(
                        label="Individual Scene Audio Files",
                        file_count="multiple",
                        visible=False
                    )
            
            scene_management_status = gr.Textbox(label="Scene Management Status", interactive=False)
            storyboard_status = gr.Textbox(label="Storyboard Status", interactive=False)
            
            def create_storyboard_handler(selected_scenes):
                # Extract indices from selected scene descriptions
                indices = []
                for scene_desc in selected_scenes:
                    try:
                        # Extract index from "Scene X: ..." format
                        scene_num = int(scene_desc.split(":")[0].replace("Scene ", "")) - 1
                        indices.append(scene_num)
                    except:
                        continue
                
                return studio.create_storyboard(indices)
            
            def refresh_scenes_handler():
                return (gr.update(choices=get_current_scenes()), get_scene_details())
            
            def remove_scene_handler(scene_num):
                if not studio.current_project or not studio.current_project.scenes:
                    return "❌ No scenes available in current project", get_scene_details(), gr.update(choices=get_current_scenes())
                
                if scene_num is None or scene_num == 0 or scene_num <= 0:
                    return "❌ Please enter a valid scene number (1 or greater)", get_scene_details(), gr.update(choices=get_current_scenes())
                
                try:
                    # Convert from 1-based to 0-based index
                    scene_index = int(scene_num) - 1
                    if scene_index >= len(studio.current_project.scenes):
                        return f"❌ Scene number {scene_num} doesn't exist. Project has {len(studio.current_project.scenes)} scenes.", get_scene_details(), gr.update(choices=get_current_scenes())
                    
                    result = studio.remove_scene(scene_index)
                    return result, get_scene_details(), gr.update(choices=get_current_scenes())
                except (ValueError, TypeError):
                    return "❌ Please enter a valid integer scene number", get_scene_details(), gr.update(choices=get_current_scenes())
            
            def reorder_scenes_handler(order_str):
                if not studio.current_project or not studio.current_project.scenes:
                    return "❌ No scenes available in current project", get_scene_details(), gr.update(choices=get_current_scenes())
                
                if not order_str or not order_str.strip():
                    return "❌ Please enter a new order", get_scene_details(), gr.update(choices=get_current_scenes())
                
                try:
                    # Parse comma-separated numbers and convert from 1-based to 0-based
                    order_1based = [int(x.strip()) for x in order_str.split(",")]
                    
                    # Validate the order
                    max_scenes = len(studio.current_project.scenes)
                    if any(x <= 0 or x > max_scenes for x in order_1based):
                        return f"❌ Scene numbers must be between 1 and {max_scenes}", get_scene_details(), gr.update(choices=get_current_scenes())
                    
                    if len(order_1based) != max_scenes:
                        return f"❌ Must include all {max_scenes} scenes in new order", get_scene_details(), gr.update(choices=get_current_scenes())
                    
                    order_0based = [x - 1 for x in order_1based]
                    
                    result = studio.reorder_scenes(order_0based)
                    return result, get_scene_details(), gr.update(choices=get_current_scenes())
                    
                except ValueError:
                    return "❌ Invalid format. Use comma-separated numbers (e.g., 3,1,2,4)", get_scene_details(), gr.update(choices=get_current_scenes())
            
            def auto_generate_dialogue_handler(selected_scenes):
                """Auto-generate dialogue text for selected storyboard scenes"""
                if not studio.current_project or not selected_scenes:
                    return "❌ Please select scenes first"
                
                try:
                    # Extract scene indices
                    indices = []
                    for scene_desc in selected_scenes:
                        scene_num = int(scene_desc.split(":")[0].replace("Scene ", "")) - 1
                        indices.append(scene_num)
                    
                    all_dialogue = []
                    dialogue_count = 0
                    
                    for idx in indices:
                        if idx < len(studio.current_project.scenes):
                            scene = studio.current_project.scenes[idx]
                            
                            # If scene has no characters, use project characters
                            scene_characters = scene.characters if scene.characters else list(studio.current_project.character_ids)
                            
                            if not scene_characters:
                                continue
                                
                            # Generate dialogue for this scene
                            dialogue_lines = studio.generate_scene_dialogue_suggestions(
                                scene.instruction, 
                                scene_characters
                            )
                            
                            dialogue_count += len(dialogue_lines)
                            
                            if dialogue_lines:
                                # Add scene header
                                all_dialogue.append(f"\n--- Scene {idx+1}: {scene.instruction[:50]}... ---")
                                # Convert to text format
                                for line in dialogue_lines:
                                    character = line.get('character', 'Unknown')
                                    text = line.get('text', '')
                                    all_dialogue.append(f"{character}: {text}")
                    
                    if dialogue_count > 0:
                        dialogue_text = "\n".join(all_dialogue)
                        return f"✅ Generated {dialogue_count} dialogue lines for {len(indices)} scenes", gr.update(value=dialogue_text, visible=True)
                    else:
                        return "❌ No dialogue generated (check API key and scene/character data)", gr.update(visible=False)
                        
                except Exception as e:
                    return f"❌ Error: {str(e)}", gr.update(visible=False)
            
            def generate_storyboard_audio_handler(selected_scenes, dialogue_text):
                """Generate complete audio package for storyboard using generated dialogue"""
                if not studio.current_project or not selected_scenes:
                    return "❌ Please select scenes first", gr.update(visible=False), gr.update(visible=False)
                
                if not dialogue_text.strip():
                    return "❌ Please generate dialogue first using 'Auto-Generate Scene Dialogue'", gr.update(visible=False), gr.update(visible=False)
                
                try:
                    # Parse dialogue text back to usable format
                    all_audio_files = []
                    dialogue_lines = []
                    current_character = ""
                    
                    for line in dialogue_text.split('\n'):
                        line = line.strip()
                        if not line:
                            continue
                        
                        if line.startswith('---'):  # Scene headers
                            continue
                            
                        if ':' in line:
                            character, text = line.split(':', 1)
                            character = character.strip()
                            text = text.strip()
                            
                            if character and text:
                                dialogue_lines.append({
                                    'character': character,
                                    'text': text
                                })
                    
                    if not dialogue_lines:
                        return "❌ No valid dialogue found to generate audio", gr.update(visible=False), gr.update(visible=False)
                    
                    # Generate audio for all dialogue
                    for line in dialogue_lines:
                        success, audio_path, message = studio.voice_studio.generate_character_dialogue(
                            line['character'], line['text']
                        )
                        if success and audio_path:
                            all_audio_files.append(str(audio_path))
                    
                    if all_audio_files:
                        # Stitch audio files together
                        stitch_message, stitched_path = studio.stitch_audio_files(all_audio_files, "storyboard_audio")
                        
                        if stitched_path:
                            return (f"✅ Generated and stitched {len(all_audio_files)} audio files into complete storyboard", 
                                   gr.update(value=stitched_path, visible=True),
                                   gr.update(value=all_audio_files, visible=True))
                        else:
                            return (f"⚠️ Generated {len(all_audio_files)} audio files but stitching failed: {stitch_message}", 
                                   gr.update(visible=False),
                                   gr.update(value=all_audio_files, visible=True))
                    else:
                        return "❌ No audio generated (check ElevenLabs API key)", gr.update(visible=False), gr.update(visible=False)
                        
                except Exception as e:
                    return f"❌ Error: {str(e)}", gr.update(visible=False), gr.update(visible=False)
            
            create_storyboard_btn.click(
                create_storyboard_handler,
                inputs=[scenes_checklist],
                outputs=[storyboard_status, storyboard_output]
            )
            
            auto_generate_dialogue_btn.click(
                auto_generate_dialogue_handler,
                inputs=[scenes_checklist],
                outputs=[storyboard_audio_status, storyboard_dialogue_display]
            )
            
            generate_storyboard_audio_btn.click(
                generate_storyboard_audio_handler,
                inputs=[scenes_checklist, storyboard_dialogue_display],
                outputs=[storyboard_audio_status, storyboard_stitched_player, generated_storyboard_audio]
            )
            
            refresh_scenes_btn.click(
                refresh_scenes_handler,
                outputs=[scenes_checklist, scene_details]
            )
            
            remove_scene_btn.click(
                remove_scene_handler,
                inputs=[scene_to_remove],
                outputs=[scene_management_status, scene_details, scenes_checklist]
            )
            
            reorder_btn.click(
                reorder_scenes_handler,
                inputs=[reorder_input],
                outputs=[scene_management_status, scene_details, scenes_checklist]
            )
            
            # Fix for scene selection synchronization when switching projects
            load_project_btn.click(
                lambda: gr.update(choices=get_current_scenes()),
                outputs=[scenes_checklist]
            )
        
        # =============================================================================
        # VOICE & AUDIO TAB
        # =============================================================================
        
        with gr.Tab("🎵 Voice & Audio") as audio_tab:
            gr.Markdown("### Character Voice Management & Scene Audio Generation")
            
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("#### Voice Profile Management")
                    
                    # Available voices
                    refresh_voices_btn = gr.Button("🔄 Refresh Available Voices", variant="secondary")
                    voices_dropdown = gr.Dropdown(label="Available Voices", choices=[], interactive=True)
                    
                    # Character voice assignment
                    voice_char_dropdown = gr.Dropdown(
                        label="Assign Voice to Character",
                        choices=[(c.name, c.id) for c in studio.data_manager.list_characters()],
                        interactive=True
                    )
                    assign_voice_btn = gr.Button("Assign Voice to Character", variant="primary")
                    
                    # Voice profile status
                    voice_profiles_display = gr.Markdown("**No voice profiles yet**")
                    voice_status = gr.Textbox(label="Voice Status", interactive=False)
                    
                    # Audio Testing - moved here from below
                    gr.Markdown("#### Audio Testing")
                    test_dialogue_input = gr.Textbox(
                        label="Test Dialogue",
                        placeholder="Enter test text to generate with selected character voice...",
                        lines=2
                    )
                    test_character_dropdown = gr.Dropdown(
                        label="Test Character Voice",
                        choices=[(c.name, c.id) for c in studio.data_manager.list_characters()],
                        interactive=True
                    )
                    test_dialogue_btn = gr.Button("🎤 Test Character Voice", variant="secondary")
                    test_audio_status = gr.Textbox(label="Test Status", interactive=False)
                    
                    # Audio playback and download for test audio
                    with gr.Group():
                        gr.Markdown("**Test Audio Player:**")
                        test_audio_player = gr.Audio(label="🎧 Play Test Audio", visible=False)
                        test_audio_file = gr.File(label="📥 Download Test Audio", visible=False)
                
                with gr.Column(scale=2):
                    gr.Markdown("#### Scene Dialogue Generation")
                    
                    # Scene dialogue generator
                    scene_desc_input = gr.Textbox(
                        label="Scene Description",
                        placeholder="Describe the scene (e.g., 'The heroes discuss their battle plan')",
                        lines=3
                    )
                    
                    scene_characters_checklist = gr.CheckboxGroup(
                        label="Characters in Scene",
                        choices=[(c.name, c.id) for c in studio.data_manager.list_characters()],
                        interactive=True
                    )
                    
                    generate_dialogue_btn = gr.Button("🎭 Generate Scene Dialogue", variant="primary")
                    
                    # Dialogue editor - more user-friendly format
                    dialogue_editor = gr.Textbox(
                        label="Scene Dialogue (Edit as needed)",
                        placeholder="Generated dialogue will appear here. Edit each line as:\nCharacter Name: Dialogue text\n\nExample:\nHero: We need to find a way out of here!\nVillain: There is no escape!",
                        lines=10,
                        interactive=True
                    )
                    
                    # Audio generation
                    generate_audio_btn = gr.Button("🔊 Generate Scene Audio", variant="secondary")
                    audio_generation_status = gr.Textbox(label="Audio Status", interactive=False)
                    
                    # Audio files display with playback
                    with gr.Group():
                        gr.Markdown("**Scene Audio Files:**")
                        generated_audio_player = gr.Audio(label="🎧 Play Scene Audio", visible=False)
                        generated_audio_files = gr.File(
                            label="📥 Download Scene Audio Files",
                            file_count="multiple",
                            visible=False
                        )
                        
                        # Audio stitching
                        stitch_scene_audio_btn = gr.Button("🔗 Stitch Audio Files", variant="secondary", visible=False)
                        stitched_scene_player = gr.Audio(label="🎧 Play Stitched Audio", visible=False)
            
                    # Add Storyboard Voiceover to second column
                    gr.Markdown("#### Storyboard Voiceover")
                    voiceover_text_input = gr.Textbox(
                        label="Storyboard Narration",
                        placeholder="Enter narration text for your storyboard...",
                        lines=4
                    )
                    generate_voiceover_btn = gr.Button("🎙️ Generate Voiceover", variant="secondary")
                    voiceover_status = gr.Textbox(label="Voiceover Status", interactive=False)
                    
                    # Voiceover audio playback and download
                    with gr.Group():
                        gr.Markdown("**Voiceover Player:**")
                        voiceover_audio_player = gr.Audio(label="🎧 Play Voiceover", visible=False)
                        voiceover_file = gr.File(label="📥 Download Voiceover", visible=False)
            
            # Sound Effects and Music Generation Section
            with gr.Row():
                with gr.Column():
                    gr.Markdown("#### Sound Effects & Music Generation")
                    gr.Markdown("*Coming Soon: AI-powered sound effects and background music generation*")
                    
                    sfx_description = gr.Textbox(
                        label="Sound Effect Description",
                        placeholder="Describe the sound effect you need (e.g., 'thunder storm', 'footsteps on gravel')",
                        lines=2
                    )
                    generate_sfx_btn = gr.Button("🔊 Generate Sound Effect", variant="secondary", interactive=False)
                    
                    music_style = gr.Textbox(
                        label="Music Style",
                        placeholder="Describe the background music (e.g., 'epic orchestral battle music', 'calm acoustic guitar')",
                        lines=2
                    )
                    generate_music_btn = gr.Button("🎵 Generate Background Music", variant="secondary", interactive=False)
            
            # Voice management event handlers
            def refresh_voices_handler():
                voices = studio.get_available_voices()
                voice_choices = [(f"{v.get('name', 'Unknown')} ({v.get('category', 'N/A')})", v.get('voice_id', '')) 
                               for v in voices]
                profiles = studio.get_character_voice_profiles()
                profiles_text = "**Current Voice Profiles:**\n\n" if profiles else "**No voice profiles yet**"
                for profile in profiles:
                    profiles_text += f"• {profile['character']}: {profile['voice']}\n"
                return gr.update(choices=voice_choices), profiles_text
            
            def assign_voice_handler(char_id, voice_selection):
                if not char_id or not voice_selection:
                    return "❌ Please select both character and voice", gr.update()
                
                voices = studio.get_available_voices()
                voice_data = next((v for v in voices if v['voice_id'] == voice_selection), None)
                if not voice_data:
                    return "❌ Selected voice not found", gr.update()
                
                result = studio.create_character_voice_profile(char_id, voice_selection, voice_data['name'])
                
                # Refresh profiles display
                profiles = studio.get_character_voice_profiles()
                profiles_text = "**Current Voice Profiles:**\n\n" if profiles else "**No voice profiles yet**"
                for profile in profiles:
                    profiles_text += f"• {profile['character']}: {profile['voice']}\n"
                
                return result, profiles_text
            
            def generate_dialogue_handler(scene_desc, character_ids):
                if not scene_desc or not character_ids:
                    return "❌ Please provide scene description and select characters", ""
                
                dialogue_lines = studio.generate_scene_dialogue_suggestions(scene_desc, character_ids)
                # Convert JSON format to user-friendly text format
                dialogue_text = ""
                for line in dialogue_lines:
                    character = line.get('character', 'Unknown')
                    text = line.get('text', '')
                    dialogue_text += f"{character}: {text}\n"
                
                return "✅ Dialogue generated! Edit as needed below.", dialogue_text
            
            def generate_scene_audio_handler(scene_desc, dialogue_text):
                if not scene_desc or not dialogue_text:
                    return "❌ Please provide scene description and dialogue", gr.update(visible=False)
                
                try:
                    # Parse text format back to JSON for audio generation
                    dialogue_lines = []
                    for line in dialogue_text.strip().split('\n'):
                        if ':' in line:
                            character, text = line.split(':', 1)
                            dialogue_lines.append({
                                'character': character.strip(),
                                'text': text.strip()
                            })
                    
                    if not dialogue_lines:
                        return "❌ No valid dialogue lines found. Please use format 'Character: Dialogue'", gr.update(visible=False)
                    
                    message, scene_data = studio.generate_scene_audio(scene_desc, dialogue_lines)
                    
                    if scene_data:
                        audio_files = [line["audio_file"] for line in scene_data["dialogue_files"]]
                        # Use first audio file for the player, all files for download
                        first_audio = audio_files[0] if audio_files else None
                        show_stitch_btn = len(audio_files) > 1  # Only show stitch button if multiple files
                        return (message, 
                               gr.update(value=first_audio, visible=True) if first_audio else gr.update(visible=False),
                               gr.update(value=audio_files, visible=True),
                               gr.update(visible=show_stitch_btn))
                    else:
                        return (message, 
                               gr.update(visible=False), 
                               gr.update(visible=False),
                               gr.update(visible=False))
                        
                except Exception as e:
                    return (f"❌ Error: {str(e)}", 
                           gr.update(visible=False), 
                           gr.update(visible=False),
                           gr.update(visible=False))
            
            def generate_voiceover_handler(voiceover_text):
                if not voiceover_text.strip():
                    return ("❌ Please enter voiceover text", 
                           gr.update(visible=False), gr.update(visible=False))
                
                message, audio_path = studio.generate_storyboard_voiceover(voiceover_text.strip())
                if audio_path:
                    return (message, 
                           gr.update(value=str(audio_path), visible=True),  # Audio player
                           gr.update(value=audio_path, visible=True))       # File download
                return (message, 
                       gr.update(visible=False), 
                       gr.update(visible=False))
            
            def test_dialogue_handler(char_id, test_text):
                if not char_id or not test_text.strip():
                    return ("❌ Please select character and enter test text", 
                           gr.update(visible=False), gr.update(visible=False))
                
                message, audio_path = studio.generate_character_dialogue(char_id, test_text.strip())
                if audio_path:
                    return (message, 
                           gr.update(value=str(audio_path), visible=True),  # Audio player
                           gr.update(value=audio_path, visible=True))       # File download
                return (message, 
                       gr.update(visible=False), 
                       gr.update(visible=False))
            
            def stitch_scene_audio_handler(audio_files):
                """Stitch multiple scene audio files together"""
                if not audio_files or len(audio_files) < 2:
                    return "❌ Need at least 2 audio files to stitch", gr.update(visible=False)
                
                message, stitched_path = studio.stitch_audio_files(audio_files, "scene_audio")
                
                if stitched_path:
                    return message, gr.update(value=stitched_path, visible=True)
                else:
                    return message, gr.update(visible=False)
            
            # Bind voice management events
            refresh_voices_btn.click(
                refresh_voices_handler,
                outputs=[voices_dropdown, voice_profiles_display]
            )
            
            assign_voice_btn.click(
                assign_voice_handler,
                inputs=[voice_char_dropdown, voices_dropdown],
                outputs=[voice_status, voice_profiles_display]
            )
            
            generate_dialogue_btn.click(
                generate_dialogue_handler,
                inputs=[scene_desc_input, scene_characters_checklist],
                outputs=[audio_generation_status, dialogue_editor]
            )
            
            generate_audio_btn.click(
                generate_scene_audio_handler,
                inputs=[scene_desc_input, dialogue_editor],
                outputs=[audio_generation_status, generated_audio_player, generated_audio_files, stitch_scene_audio_btn]
            )
            
            stitch_scene_audio_btn.click(
                stitch_scene_audio_handler,
                inputs=[generated_audio_files],
                outputs=[audio_generation_status, stitched_scene_player]
            )
            
            generate_voiceover_btn.click(
                generate_voiceover_handler,
                inputs=[voiceover_text_input],
                outputs=[voiceover_status, voiceover_audio_player, voiceover_file]
            )
            
            test_dialogue_btn.click(
                test_dialogue_handler,
                inputs=[test_character_dropdown, test_dialogue_input],
                outputs=[test_audio_status, test_audio_player, test_audio_file]
            )
        
        # =============================================================================
        # EXPORT HUB TAB
        # =============================================================================
        
        with gr.Tab("💾 Export Hub") as export_tab:
            gr.Markdown("### Export Project Data")
            
            export_project_btn = gr.Button("Export Current Project", variant="primary")
            export_characters_btn = gr.Button("Export All Characters")
            export_everything_btn = gr.Button("Export Everything")
            
            export_status = gr.Textbox(label="Export Status", interactive=False)
            download_info = gr.Markdown()
            
            def export_project_handler():
                if not studio.current_project:
                    return "❌ No current project to export"
                
                # Create export data
                export_data = {
                    "project": asdict(studio.current_project),
                    "export_timestamp": datetime.now().isoformat(),
                    "export_type": "project"
                }
                
                # Save to exports directory
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                export_file = EXPORTS_DIR / f"project_{studio.current_project.name}_{timestamp}.json"
                
                with open(export_file, 'w') as f:
                    json.dump(export_data, f, indent=2)
                
                return f"✅ Project exported to {export_file.name}"
            
            export_project_btn.click(
                export_project_handler,
                outputs=[export_status]
            )
        
        # =============================================================================
        # TAB SELECTION HANDLERS
        # =============================================================================
        
        # Tab selection handlers to update current tab indicator
        api_tab.select(lambda: "🔑 API Setup", outputs=[current_tab])
        project_tab.select(lambda: "📁 Project Manager", outputs=[current_tab])
        char_dev_tab.select(lambda: "💬 Character Development", outputs=[current_tab])
        char_lib_tab.select(lambda: "👥 Character Library", outputs=[current_tab])
        img_gen_tab.select(lambda: "🎨 Image Generation", outputs=[current_tab])
        chat_img_tab.select(lambda: "💬 Chat Image Generation", outputs=[current_tab])
        scene_tab.select(lambda: "🎬 Scene Creator", outputs=[current_tab])
        mask_tab.select(lambda: "✏️ Mask Editor", outputs=[current_tab])
        expansion_tab.select(lambda: "📐 Image Expansion", outputs=[current_tab])
        storyboard_tab.select(lambda: "📖 Storyboard Builder", outputs=[current_tab])
        audio_tab.select(lambda: "🎵 Voice & Audio", outputs=[current_tab])
        export_tab.select(lambda: "💾 Export Hub", outputs=[current_tab])
        
        return app

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    print("""
    🎨 Comprehensive Character Studio & Storyboard Generator
    
    📋 Features:
    ✅ AI-powered character development chat
    ✅ Persistent character library with reference images
    ✅ Multi-project management with auto-save
    ✅ Advanced image generation with Gemini 2.5 Flash Image
    ✅ Multi-character scene creation
    ✅ Storyboard assembly and export
    ✅ ElevenLabs voice integration with character dialogue
    ✅ Scene audio generation and storyboard voiceover
    ✅ Complete audio-visual storyboard production
    
    🔧 Required Dependencies:
    pip install gradio pillow google-generativeai requests
    
    🔑 Setup:
    1. Get your Gemini API key from https://ai.google.dev/
    2. Get your ElevenLabs API key from https://elevenlabs.io/
    3. Enter both keys in the API Setup tab
    4. Start creating characters, projects, and audio!
    
    🚀 Starting application...
    """)
    
    app = create_gradio_interface()
    app.launch(
        share=False,
        debug=True,
        show_error=True
    )