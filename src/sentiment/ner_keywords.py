# extracts named entities and keywords from a Polymarket question string
# used by all sentiment collection scripts to filter source text by relevance
#
# returns a ranked list of keywords: named entities first, then proper nouns
# downstream scripts should take top-N to avoid over-broad matching
#
# keywords = extract_keywords("Will Iran attack Israel before June 2025?") -> ["Iran", "Israel"]
# to add new names/places: edit src/sentiment/geo_vocab_mapping.py
 
import re # for regex operations
import spacy
from src.sentiment.geo_vocab_mapping import KNOWN_NAMES, STOPWORDS
 

nlp = spacy.load("en_core_web_sm")

# entity types worth keeping for geopolitical markets
# GPE = countries/cities, PERSON = named people, ORG = organisations, NORP = nationalities/groups
KEEP_ENTITY_TYPES = {"GPE", "PERSON", "ORG", "NORP"}
 
 
def _normalise(text):
    return KNOWN_NAMES.get(text.lower(), text)
 
def extract_keywords(question, max_keywords = 6):
    # prepend neutral prefix so names at position 0 aren't misread as verbs
    # e.g. "Xi Jinping out in 2025?" -> "About: Xi Jinping out in 2025?"
    doc = nlp("About: " + question)
 
    seen = set()
    keywords = []
 
    def add(word):
        clean = word.strip()
        if not clean:
            return
        if clean.lower() in STOPWORDS:
            return
        if clean.lower() in seen:
            return
        # allow 2-char terms (EU, UK) but skip single chars
        if len(clean) < 2:
            return
        seen.add(clean.lower())
        keywords.append(clean)
 
    # 1. named entities, normalised to canonical short form
    for ent in doc.ents:
        if ent.label_ in KEEP_ENTITY_TYPES:
            # strip leading "will " because spaCy sometimes bundles it into the entity
            text = re.sub(r"(?i)^will\s+", "", ent.text).strip()
            add(_normalise(text))
 
    # 2. proper nouns not caught by NER
    for token in doc:
        if token.pos_ == "PROPN" and not token.ent_type_:
            if token.text.lower() == "will" and token.i == 0:
                continue
            add(_normalise(token.text))
 
    # 3. token-level KNOWN_NAMES fallback
    # catches names spaCy doesn't recognise at all (not in its vocabulary)
    # POS guard skips pronouns/verbs so "us" as a pronoun doesn't map to "USA"
    for token in doc:
        if token.pos_ in ("VERB", "PRON", "ADP", "DET", "CCONJ", "SCONJ"):
            continue
        canonical = KNOWN_NAMES.get(token.text.lower())
        if canonical and canonical.lower() not in seen:
            seen.add(canonical.lower())
            keywords.append(canonical)
 
    return keywords[:max_keywords]
 
 
def keywords_to_pattern(keywords):
    if not keywords:
        return re.compile(r"(?!)")  # never matches
    escaped = [re.escape(kw) for kw in keywords]
    return re.compile("|".join(escaped), re.IGNORECASE)
