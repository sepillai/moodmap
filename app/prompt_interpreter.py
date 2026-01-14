import numpy as np

GLOVE_PATH = "embeddings/glove.6B.100d.txt"


def load_glove_embeddings():
    """Load GloVe embeddings, return empty dict if file not found."""
    embeddings = {}
    try:
        with open(GLOVE_PATH, "r", encoding="utf8") as f:
            for line in f:
                values = line.split()
                word = values[0]
                vector = np.asarray(values[1:], dtype="float32")
                embeddings[word] = vector
    except FileNotFoundError:
        print(f"Warning: GloVe embeddings not found at {GLOVE_PATH}")
        print("Keyword-based interpretation will be limited. ML model will still work.")
    except Exception as e:
        print(f"Warning: Error loading GloVe embeddings: {e}")
    return embeddings


glove_embeddings = load_glove_embeddings()


def embed(word):
    return glove_embeddings.get(word.lower(), None)


PARAM_KEYWORDS = {
    "speed_up": [
        "fast", "energetic", "upbeat", "intense",
        "driving", "high energy", "rapid tempo"
    ],
    "slow_down": [
        "slow", "chill", "relaxed", "smooth",
        "downtempo", "ambient pace", "laid back"
    ],
    "brighten": [
        "bright", "airy", "sparkly", "light",
        "shiny", "crisp high end", "vibrant tone"
    ],
    "darken": [
        "dark", "moody", "warm", "deep",
        "low fidelity", "brooding tone", "dusty"
    ],
    "bass_up": [
        "bass", "deep", "low", "sub",
        "heavy low end", "thick bass", "boomy"
    ],
    "reverb_up": [
        "echo", "space", "ambient",
        "cinematic atmosphere", "wide hall", "dreamy reverb"
    ],
    "compress": [
        "punchy", "tight", "snappy",
        "aggressive compression", "club-ready", "radio loudness"
    ]
}


GENRE_PRESETS = {
    "lofi": {
        "tempo_factor": 0.9,
        "brightness_db": -4,
        "bass_db": 3,
        "reverb": 0.2,
        "compression": 0.1
    },
    "edm": {
        "tempo_factor": 1.15,
        "brightness_db": 6,
        "bass_db": 6,
        "reverb": 0.3,
        "compression": 0.8
    },
    "trap": {
        "tempo_factor": 1.05,
        "brightness_db": -2,
        "bass_db": 8,
        "reverb": 0.1,
        "compression": 0.7
    },
    "house": {
        "tempo_factor": 1.12,
        "brightness_db": 5,
        "bass_db": 4,
        "reverb": 0.25,
        "compression": 0.6
    },
    "ambient": {
        "tempo_factor": 0.8,
        "brightness_db": 2,
        "bass_db": -1,
        "reverb": 0.8,
        "compression": 0.0
    },
    "cinematic": {
        "tempo_factor": 1.0,
        "brightness_db": 3,
        "bass_db": 3,
        "reverb": 0.7,
        "compression": 0.4
    },
    "rock": {
        "tempo_factor": 1.05,
        "brightness_db": 4,
        "bass_db": 2,
        "reverb": 0.15,
        "compression": 0.6
    },
    "jazz": {
        "tempo_factor": 0.95,
        "brightness_db": 1,
        "bass_db": 2,
        "reverb": 0.3,
        "compression": 0.2
    }
}


def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def embed_phrase(phrase: str):
    words = phrase.lower().split()
    vectors = [embed(w) for w in words if embed(w) is not None]

    if not vectors:
        return None

    return np.mean(vectors, axis=0)


def interpret_prompt(prompt: str):
    words = prompt.lower().split()

    phrases = []
    for i in range(len(words)):
        if i + 1 < len(words):
            phrases.append(words[i] + " " + words[i+1])
        if i + 2 < len(words):
            phrases.append(words[i] + " " + words[i+1] + " " + words[i+2])

    category_scores = {cat: 0.0 for cat in PARAM_KEYWORDS}

    # Score single words
    for w in words:
        w_vec = embed(w)
        if w_vec is None:
            continue

        for cat, kw_list in PARAM_KEYWORDS.items():
            sims = []
            for kw in kw_list:
                kw_vec = embed_phrase(kw) if " " in kw else embed(kw)
                sims.append(cosine_similarity(w_vec, kw_vec))
            if sims:
                category_scores[cat] += max(sims)

    # Score multi-word phrases (weighted slightly higher)
    for ph in phrases:
        ph_vec = embed_phrase(ph)
        if ph_vec is None:
            continue

        for cat, kw_list in PARAM_KEYWORDS.items():
            sims = []
            for kw in kw_list:
                kw_vec = embed_phrase(kw) if " " in kw else embed(kw)
                sims.append(cosine_similarity(ph_vec, kw_vec))
            if sims:
                category_scores[cat] += max(sims) * 1.5

    return category_scores


def normalize_scores(scores):
    max_score = max(scores.values()) or 1
    return {k: (v / max_score if max_score != 0 else 0.0) for k, v in scores.items()}


def scores_to_audio_params(scores):
    s = normalize_scores(scores)

    return {
        "tempo_factor": 1.0 + (s["speed_up"] * 0.15) - (s["slow_down"] * 0.20),
        "brightness_db": (s["brighten"] * 8) - (s["darken"] * 8),
        "bass_db": s["bass_up"] * 6,
        "reverb": s["reverb_up"] * 0.5,
        "compression": s["compress"] * 0.5
    }


def detect_genre(prompt: str):
    text = prompt.lower()
    for genre in GENRE_PRESETS.keys():
        if genre in text:
            return genre
    return None


def interpret(prompt: str, use_ml: bool = True):
    """
    Full interpretation pipeline with ML model fallback to keyword system.
    
    1. Try ML model prediction (if available)
    2. Fallback to keyword-based interpretation
    3. Detect genre preset (if any)
    4. Apply genre preset overrides if a genre was mentioned
    
    Args:
        prompt: Text prompt
        use_ml: Whether to try ML model first (default: True)
    
    Returns:
        Dictionary of audio parameters
    """
    # Try ML model first (if available and enabled)
    if use_ml:
        try:
            from app.ml.model_loader import get_model_loader
            loader = get_model_loader()
            if loader.is_loaded:
                ml_params = loader.predict(prompt)
                if ml_params:
                    # Still apply genre overrides if detected
                    genre = detect_genre(prompt)
                    if genre:
                        genre_params = GENRE_PRESETS[genre]
                        for k, v in genre_params.items():
                            ml_params[k] = v
                    return ml_params
        except Exception:
            # Fall back to keyword system if ML fails
            pass
    
    # Fallback: keyword-based interpretation
    # 1. genre detection
    genre = detect_genre(prompt)

    scores = interpret_prompt(prompt)

    params = scores_to_audio_params(scores)

    if genre:
        genre_params = GENRE_PRESETS[genre]
        for k, v in genre_params.items():
            params[k] = v

    return params
