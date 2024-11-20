# Configuration for Whisper models and languages
whisper_model_config = {
    "default_transcribe": {
        "model_id": "openai/whisper-large-v3-turbo",
        "config": "openai/whisper-large-v3-turbo",
        "dtype": "float16"
    },
    "default_translate": {
        "model_id": "openai/whisper-large-v2",
        "config": "openai/whisper-large-v2",
        "dtype": "float32"
    },
    "large-v3": {
        "model_id": "openai/whisper-large-v3",
        "config": "openai/whisper-large-v3",
        "dtype": "float16"
    },
    "large-v2": {
        "model_id": "openai/whisper-large-v2",
        "config": "openai/whisper-large-v2",
        "dtype": "float32"
    },
    "medium": {
        "model_id": "openai/whisper-medium",
        "config": "openai/whisper-medium",
        "dtype": "float32"
    },
    "small": {
        "model_id": "openai/whisper-small",
        "config": "openai/whisper-small",
        "dtype": "float32"
    },
    "base": {
        "model_id": "openai/whisper-base",
        "config": "openai/whisper-base",
        "dtype": "float32"
    },
    "tiny": {
        "model_id": "openai/whisper-tiny",
        "config": "openai/whisper-tiny",
        "dtype": "float32"
    }
}
