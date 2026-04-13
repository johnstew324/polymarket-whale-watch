# vocabulary data for the NER keyword extraction pipeline
# STOPWORDS/ KNOWN NAME MAPPINGS ARE STILL BEING ADDED! FIX IF FIND ANYTHING 


# all entries must be lowercase the check does clean.lower() in STOPWORDS
STOPWORDS = {
    # prediction market boilerplate
    "yes", "no", "will", "market", "polymarket", "resolve", "resolves",
    "before", "after", "end", "start", "date", "price", "week", "month",
    "year", "day", "time",

    # months 
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    # quarters 
    "q1", "q2", "q3", "q4",
    "2022", "2023", "2024", "2025", "2026",
    # ordinals 
    "first", "second", "third", "fourth", "new", "old",

    # titles and roles
    "prime", "minister", "president", "secretary", "general", "leader",
    "chief", "chairman", "director", "ambassador", "senator", "governor",
    "supreme",
    # noise that leaks through as proper nouns
    "et",
    "next", "hell", "hottest", "god", "wars", "div",
    "ass", "brutal", "brutally", "beautiful", "deadlock", "dictator",
    "globalist", "globalism", "suit", "pond", "roof", "egg",
    "trillion", "victory", "edition", "phase", "borders", "powers",
    "slander", "natural", "million", "emergency", "response", "rooms",
    "people", "power", "world", "climate", "economic", "development",
    "military", "human", "shield", "proxy", "war", "oblast",
    "da", "diy", "onseptember", "clock", "pay", "pm", "country",

    "nord", "stream",  # once consumed fix is in, these shouldn't appear but belt-and-braces
    "iron", "dome",
    "suez", "canal",
    "hostage", "aircraft", "farmer", "heritage",
    "joe",   # Biden fragment
    "steve", # no last name
    "marco", # Rubio fragment - handle via KNOWN_NAMES instead
    "javier", # Milei fragment
    "recep", "tayyip",  # Erdoğan fragments
    "luiz", "inácio", "silva",  # Lula fragments
    "nord", "stream",  # once consumed fix is in, these shouldn't appear but belt-and-braces
    "iron", "dome",
    "suez", "canal", "chancellor", "sovereign", "sovereignty",
}

# maps lowercase token/phrase  canonical display form
# two purposes:
# pass 1 normalisation - collapses "Benjamin Netanyahu" -> "Netanyahu" etc.
# pass 3 fallback - catches names spaCy doesn't recognise at all
# all keys must be lowercase

KNOWN_NAMES = {

    # people 

    # zelensky
    "zelenskyy":                        "Zelenskyy",
    "zelensky":                         "Zelenskyy",
    "volodymyr zelenskyy":              "Zelenskyy",
    "volodymyr zelensky":               "Zelenskyy",
    "volodymyr":                        "Zelenskyy",

    # netanyahu
    "netanyahu":                        "Netanyahu",
    "benjamin netanyahu":               "Netanyahu",
    "benjamin":                         "Netanyahu",

    # putin
    "putin":                            "Putin",
    "vladimir putin":                   "Putin",

    # trump
    "trump":                            "Trump",
    "donald trump":                     "Trump",

    # xi jinping
    "xi jinping":                       "Xi Jinping",
    "jinping":                          "Xi Jinping",
    "xi":                               "Xi Jinping",

    # macron
    "macron":                           "Macron",
    "emmanuel macron":                  "Macron",

    # erdogan
    "erdogan":                          "Erdoğan",
    "erdoğan":                          "Erdoğan",
    "recep tayyip erdoğan":             "Erdoğan",
    "recep tayyip erdogan":             "Erdoğan",

    # modi
    "modi":                             "Modi",
    "narendra modi":                    "Modi",

    # assad
    "assad":                            "Assad",
    "bashar al-assad":                  "Assad",

    # yoon
    "yoon":                             "Yoon",
    "yoon suk yeol":                    "Yoon",

    # lukashenko
    "lukashenko":                       "Lukashenko",
    "aleksandr lukashenko":             "Lukashenko",

    # khamenei
    "khamenei":                         "Khamenei",
    "ali khamenei":                     "Khamenei",
    "ayatollah khamenei":               "Khamenei",

    # maduro
    "maduro":                           "Maduro",
    "nicolás maduro":                   "Maduro",

    # milei
    "milei":                            "Milei",
    "javier milei":                     "Milei",

    # lula
    "lula":                             "Lula",
    "luiz inácio lula da silva":        "Lula",
    "luiz inácio lula da silva":        "Lula",
    "lula da silva":                    "Lula",

    # bolsonaro
    "bolsonaro":                        "Bolsonaro",
    "jair bolsonaro":                   "Bolsonaro",

    # starmer
    "starmer":                          "Starmer",
    "keir starmer":                     "Starmer",
    "keir":                             "Starmer",

    # orban
    "orban":                            "Orbán",
    "orbán":                            "Orbán",
    "viktor orbán":                     "Orbán",
    "viktor orban":                     "Orbán",

    # sisi
    "sissi":                            "Sisi",
    "abdel fattah el-sisi":             "Sisi",
    "abdel fattah el sisi":             "Sisi",

    # mbs
    "mbs":                              "MBS",
    "mohammed bin salman":              "MBS",
    "crown prince mohammed bin salman": "MBS",

    # nasrallah
    "nasrallah":                        "Nasrallah",
    "hassan nasrallah":                 "Nasrallah",

    # scholz
    "scholz":                           "Scholz",
    "olaf scholz":                      "Scholz",

    # merz
    "merz":                             "Merz",
    "friedrich merz":                   "Merz",

    # meloni
    "meloni":                           "Meloni",
    "giorgia meloni":                   "Meloni",

    # vance
    "vance":                            "Vance",
    "jd vance":                         "Vance",

    # witkoff
    "witkoff":                          "Witkoff",

    # rutte
    "rutte":                            "Rutte",
    "mark rutte":                       "Rutte",

    # al-sharaa
    "al-sharaa":                        "Al-Sharaa",
    "ahmed al-sharaa":                  "Al-Sharaa",
    "ahmed al shara":                   "Al-Sharaa",

    # elon musk
    "elon":                             "Elon Musk",
    "elon musk":                        "Elon Musk",
    "musk":                             "Elon Musk",

    # von der leyen
    "ursula von der leyen":             "Von der Leyen",
    "ursula":                           "Von der Leyen",
    "leyen":                            "Von der Leyen",
    "ursula von der":           "Von der Leyen",

    # pope
    "pope":                             "Pope",
    "pope leo":                         "Pope",
    "pope leo xiv":                     "Pope",
    "leo xiv":                          "Pope",
    "pope francis":                     "Pope Francis",

    # epstein
    "epstein":                          "Epstein",

    # carney
    "carney":                           "Carney",
    "mark carney":                      "Carney",

    # people missing
    "kim jong un":              "Kim Jong Un",
    "kim":                      "Kim Jong Un",   # careful, ambiguous
    "jerome powell":            "Jerome Powell",
    "powell":                   "Jerome Powell",
    "marco rubio":              "Marco Rubio",
    "rubio":                    "Marco Rubio",
    "greta thunberg":           "Greta Thunberg",
    "greta thunberg's":         "Greta Thunberg",
    "trudeau":                  "Trudeau",
    "justin trudeau":           "Trudeau",
    "obama":                    "Obama",
    "shigeru ishiba":           "Ishiba",
    "ishiba":                   "Ishiba",
    "pedro sánchez":            "Sánchez",
    "guterres":                 "Guterres",
    "antónio guterres":         "Guterres",
    "lai ching-te":             "Lai Ching-te",
    "pablo durov":              "Durov",
    "durov":                    "Durov",
    "pablo durov":               "Durov",
    "pavel durov":               "Durov",

    "mrbeast":                  "MrBeast",
    "sinwar":                   "Sinwar",
    "yahya sinwar":             "Sinwar",
    "ben gvir":                 "Ben Gvir",
    "reza pahlavi":             "Reza Pahlavi",


    "giorgia":                      "Meloni",     # showing solo
    "javier":                       "Milei",      # or just stopword it
    "marco":                        "Marco Rubio",

    


    # groups / factions
    "houthi":                           "Houthi",
    "houthis":                          "Houthi",
    "wagner":                           "Wagner",
    "idf":                              "IDF",
    "unrwa":                            "UNRWA",
    "hts":                              "HTS",
    "pkk":                              "PKK",
    "brics":                            "BRICS",

    #  country/bloc codes 
    "us":                               "USA",
    "u.s":                              "USA",
    "u.s.":                             "USA",
    "u.s.a":                            "USA",
    "usa":                              "USA",
    "united states":                    "USA",
    "uk":                               "UK",
    "united kingdom":                   "UK",
    "eu":                               "EU",
    "european union":                   "EU",
    "\"european union":                 "EU",

    "uae":                              "UAE",
    "nato":                             "NATO",

    "ukraini":                        "Ukraine",

    # demonyms → country
    "iranian":                          "Iran",
    "israeli":                          "Israel",
    "israel's":                         "Israel",
    "russian":                          "Russia",
    "syrian":                           "Syria",
    "turkish":                          "Turkey",
    "türkiye":                          "Turkey",
    "armenian":                         "Armenia",
    "lebanese":                         "Lebanon",
    "chinese":                          "China",
    "indian":                           "India",
    "south korean":                     "South Korea",
    "north korean":                     "North Korea",
    "polish":                           "Poland",
    "gazan":                            "Gaza",
    "gazans":                           "Gaza",
    "crimean":                          "Crimea",
    "saudi":                            "Saudi Arabia",

    # places spaCy misses
    "baghdad":                          "Baghdad",
    "tehran":                           "Tehran",
    "heathrow":                         "Heathrow",
    "fordow":                           "Fordow",
    "baglihar":                         "Baglihar",
    "kharg":                            "Kharg",
    "hormuz":                           "Hormuz",
    "crimea":                           "Crimea",
    "donbas":                           "Donbas",
    "donbass":                          "Donbas",
    "zaporizhzhia":                     "Zaporizhzhia",
    "mariupol":                         "Mariupol",
    "kharkiv":                          "Kharkiv",
    "pokrovsk":                         "Pokrovsk",
    "sudzha":                           "Sudzha",
    "siversk":                          "Siversk",
    "panama":                           "Panama",

    # compound place names 
    "iron dome":                        "Iron Dome",
    "nord stream":                      "Nord Stream",
    "suez canal":                       "Suez Canal",
    "black sea":                        "Black Sea",
    "persian gulf":                     "Persian Gulf",
    "south korea":                      "South Korea",
    "north korea":                      "North Korea",
    "saudi arabia":                     "Saudi Arabia",



    "artificial intelligence":  "AI",
    "a.i":                          "AI",
    "iran nuke":                "Iran",           # or keep as-is?
    "hezbollah ceasefire":      "Hezbollah",
    "ukraine ceasefire":        "Ukraine",
    "abraham accord":           "Abraham Accord",
    "warsaw pact":              "Warsaw Pact",
    "west bank":                "West Bank",
    "the west bank":            "West Bank",
    "the white house":          "White House",
    "white house":                  "White House",
    "the house of councillors": "House of Councillors",
    "the un general assembly":  "UN General Assembly",
    "un general assembly":      "UN General Assembly",





    # orgs/groups missing
    "doge":                     "DOGE",
    "icc":                      "ICC",
    "who":                      "WHO",           # careful - spaCy will grab "who" as pronoun a lot
    "wto":                      "WTO",
    "g7":                       "G7",
    "spacex":                   "SpaceX",
    "deepseek":                 "DeepSeek",
    "tesla":                    "Tesla",
    "tiktok":                   "TikTok",
    "xai":                      "xAI",
}