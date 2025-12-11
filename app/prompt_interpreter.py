import numpy as np

GLOVE_PATH = "embeddings/glove.6B.100d.txt"

def load_glove_embeddings():
    embeddings = {}
    with open(GLOVE_PATH, "r", encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype="float32")
            embeddings[word] = vector
    return embeddings

glove_embeddings = load_glove_embeddings()

def embed(word):
    return glove_embeddings.get(word.lower(), None)

PARAM_KEYWORDS = {
    "speed_up": ["fast", "energetic", "upbeat", "intense"],
    "slow_down": ["slow", "chill", "relaxed"],
    "brighten": ["bright", "airy", "sparkly"],
    "darken": ["dark", "moody", "warm"],
    "bass_up": ["bass", "deep", "low"],
    "reverb_up": ["echo", "space", "ambient"],
    "compress": ["punchy", "tight"],
}

def cosine_similarity(a, b):
    if a is None or b is None:
        return 0.0
    return np.dot(a,b) / (np.linalg.norm(a) * np.linalg.norm(b))

def interpret_prompt(prompt: str):
    words = prompt.lower().split()

    category_scores = {cat: 0.0 for cat in PARAM_KEYWORDS}

    for w in words:
        w_vec = embed(w)
        if w_vec is None:
            continue

        for cat, kw_list in PARAM_KEYWORDS.items():
            sims = []
            for kw in kw_list:
                kw_vec = embed(kw)
                sims.append(cosine_similarity(w_vec, kw_vec))
            category_scores[cat] += max(sims)
    return category_scores

def normalize_scores(scores):
    max_score = max(scores.values()) or 1
    return {k: v/max_score for k, v in scores.items()}

def scores_to_audio_params(scores):
    s = normalize_scores(scores)

    return {
        "tempo_factor": 1.0 + (s["speed_up"] * 0.15) - (s["slow_down"] * 0.20),
        "brightness_db": (s["brighten"] * 8) - (s["darken"] * 8),
        "bass_db": s["bass_up"] * 6,
        "reverb": s["reverb_up"] * 0.5,
        "compression": s["compress"] * 0.5        
    }

def interpret(prompt):
    scores = interpret_prompt(prompt)
    params = scores_to_audio_params(scores)
    return params
