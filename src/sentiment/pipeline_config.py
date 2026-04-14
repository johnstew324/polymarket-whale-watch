# vocabulary data for the NER keyword extraction pipeline
# STOPWORDS/ KNOWN NAME MAPPINGS ARE STILL BEING ADDED


# look through if anything needs to be fixed or added!


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
    "q1", "q2", "q3", "q4", "2022", "2023", "2024", "2025", "2026",
    "first", "second", "third", "fourth", "new", "old",

    # titles and roles
    "prime", "minister", "president", "secretary", "general", "leader",
    "chief", "chairman", "director", "ambassador", "senator", "governor", "supreme",
    # noise that leaks through as proper nouns
    "et", "next", "hell", "hottest", "god", "wars", "div",
    "ass", "brutal", "brutally", "beautiful", "deadlock", "dictator",
    "globalist", "globalism", "suit", "pond", "roof", "egg",
    "trillion", "victory", "edition", "phase", "borders", "powers",
    "slander", "natural", "million", "emergency", "response", "rooms",
    "people", "power", "world", "climate", "economic", "development",
    "military", "human", "shield", "proxy", "war", "oblast",
    "da", "diy", "onseptember", "clock", "pay", "pm", "country",

    "nord", "stream",  # once consumed fix is in, these shouldn't appear but belt-and-braces
    "iron", "dome", "suez", "canal", "hostage", "aircraft", "farmer", "heritage",
    "joe",   "steve", "marco", "javier", "recep", "tayyip", # fragments of already-handled names
    "luiz", "inácio", "silva",  # Lula fragments  # once consumed fix is in, these shouldn't appear but belt-and-braces
    "suez", "canal", "chancellor", "sovereign", "sovereignty",
    "artificial", "intelligence",  # caught by phrase lookup as "AI" instead
    "reza", "pahlavi",             # caught by phrase lookup
    "médecins", "sans", "frontières",
    "pavel", 
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
    "pavel":                    "Durov",

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
    "the european union":               "EU",
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
    "venezuelan":                       "Venezuela",
    "pakistani":                        "Pakistan",
    "afghan":                           "Afghanistan",

    "damascus":                     "Damascus",
    "kyiv":                         "Kyiv",
    "donetsk":                      "Donetsk",
    "persian gulf":                 "Persian Gulf",
    "iron dome":                    "Iron Dome",        # already there
    "gulf leaders summit":          "Gulf Leaders Summit",
    "nato public forum address":    "NATO Public Forum Address",
    "constitutional court":         "Constitutional Court",

    "lukashenko":                   "Lukashenko",       # already there, confirm spelling
    "maduro":                       "Maduro",           # already there
    "carney":                       "Carney", 

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
    "a.i.":                         "AI",
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







# ALL_TOPICS
# master list of all ProQuest corpora
# must match folder names exactly under data/raw/proquest/
# drives both TOPICS (parse + score) and AVAILABLE_CORPORA (collect)
# keep it alphabetical incase we want to add extra so its easy to see


ALL_TOPICS = [
    "armenia_trump",
    "bolsonaro",
    "ceasefire_russia_ukraine",
    "china_taiwan",
    "crimea",
    "elon_musk_trump",
    "gaza_israel",
    "gaza_usa",
    "germany_trump",
    "gulf_trump",
    "hamas_israel",
    "hezbollah_israel",
    "hezbollah_nasrallah",
    "houthi",
    "india_pakistan",
    "iran_israel",
    "iran_trump",
    "iran_usa",
    "iraq_israel",
    "israel_lebanon",
    "israel_saudi",
    "israel_syria",
    "israel_trump",
    "israel_yemen",
    "japan_election",
    "jerome_powell",
    "khamenei",
    "kupiansk_russia",
    "lula",
    "macron_trump",
    "mbs",
    "meloni",
    "merz_trump",
    "moscow_ukraine",
    "nato_rutte",
    "netanyahu",
    "netanyahu_unga",
    "north_korea",
    #"poland_trump", to get
    "pokrovsk_russia",
    "pope",
    "putin",
    "putin_zelenskyy",
    "russia_siversk",
    "russia_sudzha",
    "russia_syria",
    "russia_ukraine",
    "saudi_arabia_usa",
    "south_african_trump",
    "south_korea_trump",
    "starmer_trump",
    "syria",
    #"syria_usa",
    "trump_putin",
    "trump_turkey",
    "trump_un_general_assembly",
    "trump_zelenskyy",
    "usa_venezuela",
    "xi_jinping",
    "yoon",
    "zelenskyy",
]
 
# used by parse_proquest.py and finbert_scorer.py
TOPICS = ALL_TOPICS
 
# used by collect_ft.py
AVAILABLE_CORPORA = set(ALL_TOPICS)



#  4. KEYWORD_TO_CORPORA 
# maps NER keyword → list of corpora containing relevant articles
# keys must match canonical display form from KNOWN_NAMES above
 
    




KEYWORD_TO_CORPORA = {
 
    #  Israel / Palestine 

    "Israel":        ["gaza_israel", "hamas_israel", "israel_lebanon",
                      "israel_syria", "israel_yemen", "israel_saudi",
                      "netanyahu_unga", "hezbollah_israel", "iraq_israel",
                      "israel_trump", "netanyahu", "iran_israel"],

    "Gaza":          ["gaza_israel", "gaza_usa", "hamas_israel"],
    "Palestine":     ["gaza_israel", "hamas_israel", "gaza_usa"],
    "Hamas":         ["hamas_israel", "gaza_israel"],
    "Netanyahu":     ["netanyahu_unga", "hamas_israel", "netanyahu"],


    "Hezbollah":     ["hezbollah_nasrallah", "israel_lebanon", "hezbollah_israel"],
    "Nasrallah":     ["hezbollah_nasrallah"],
    "Lebanon":       ["israel_lebanon", "hezbollah_nasrallah"],
    "Houthi":        ["israel_yemen", "houthi"],
    "Saudi Arabia":  ["israel_saudi", "saudi_arabia_usa"],
    "Egypt":         ["gaza_israel", "gaza_usa"],
    "Qatar":         ["gaza_israel", "hamas_israel"],
    "IDF":           ["gaza_israel", "hamas_israel"],
    "West Bank":     ["gaza_israel", "hamas_israel"],
    "UNRWA":         ["gaza_israel", "gaza_usa"],
    "Iron Dome":     ["gaza_israel", "hamas_israel"],
    "Sinwar":        ["hamas_israel", "gaza_israel"],
    "Ben Gvir":      ["hamas_israel", "gaza_israel"],
    "Tel Aviv":      ["gaza_israel", "hamas_israel"],
    "Beirut":        ["israel_lebanon", "hezbollah_nasrallah"],
 

    # Syria 
    "Syria":         ["israel_syria", "russia_syria", "syria" ],
    "Assad":         ["israel_syria", "russia_syria", "syria"],
    "Al-Sharaa":     ["syria"],
    "HTS":           ["syria"],
    "Damascus":      ["israel_syria", "syria"],
 
    #  Russia / Ukraine
    "Russia":        ["russia_ukraine", "moscow_ukraine", "russia_syria",
                      "ceasefire_russia_ukraine", "russia_siversk", "russia_sudzha",
                      "trump_putin", "putin_zelenskyy", 
                      "kupiansk_russia", "pokrovsk_russia", "zelenskyy", "crimea"],
    "Ukraine":       ["russia_ukraine", "moscow_ukraine", "trump_zelenskyy",
                      "ceasefire_russia_ukraine", "putin_zelenskyy"
                      "russia_siversk", "russia_sudzha", "zelenskyy", "crimea"
                      ],

    "Putin":         ["trump_putin", "putin_zelenskyy", "russia_ukraine", "putin"],
    "Zelenskyy":     ["zelenskyy", "trump_zelenskyy", "putin_zelenskyy", "russia_ukraine", "ceasefire_russia_ukraine", "crimea"],

    "Ceasefire":     ["ceasefire_russia_ukraine", "russia_ukraine"],
    "Pokrovsk":      ["pokrovsk_russia"],
    "Kupiansk":      ["kupiansk_russia"],
    "Sudzha":        ["russia_sudzha"],
    "Siversk":       ["russia_siversk"],
    "Crimea":        ["crimea", "russia_ukraine", "ceasefire_russia_ukraine"],
    "Donbas":        ["russia_ukraine", "ceasefire_russia_ukraine"],
    "Donetsk":       ["russia_ukraine"],
    "Mariupol":      ["russia_ukraine"],
    "Kharkiv":       ["russia_ukraine"],
    "Zaporizhzhia":  ["russia_ukraine"],
    "Moscow":        ["moscow_ukraine"],
    "Kyiv":          ["russia_ukraine", "moscow_ukraine"],
    "Wagner":        ["russia_ukraine", "russia_syria"],
    "Black Sea":     ["russia_ukraine"],
    "Nord Stream":   ["russia_ukraine"],
    "HIMARS":        ["russia_ukraine"],
 
    #  Iran 
    "Iran":          ["iran_israel", "iran_usa", "iran_trump"],
    "Khamenei":      ["khamenei", "iran_israel", "iran_usa"],
    "Tehran":        ["iran_israel", "iran_usa", "iran_trump"],
    "Fordow":        ["iran_israel", "iran_usa"],
    "Hormuz":        ["iran_usa"],
    "Kharg":         ["iran_israel", "iran_usa"],
 
    #  Trump diplomacy 
    "Trump":         ["iran_usa", "gaza_usa", "saudi_arabia_usa","usa_venezuela"
                        "trump_zelenskyy", "trump_putin", "elon_musk_trump",
                        "germany_trump", "macron_trump", "merz_trump",
                         "south_african_trump", "south_korea_trump",
                        "starmer_trump", "trump_turkey", "trump_un_general_assembly",
                        "israel_trump", "iran_trump", "gulf_trump",
                         "armenia_trump", "nato_rutte"],

    
    "Vance":         ["trump_zelenskyy", "trump_putin"],
    "Witkoff":       ["trump_zelenskyy", "trump_putin"],
    "Elon Musk":     ["elon_musk_trump"],
    "DOGE":          ["elon_musk_trump"],
    "White House":   ["trump_zelenskyy", "trump_putin"],
    "Marco Rubio":   ["trump_zelenskyy", "trump_putin", "iran_trump"],
    "Gulf Leaders Summit": ["gulf_trump"],
    "Persian Gulf":  ["gulf_trump", "saudi_arabia_usa"],
 
    # Individual leaders 
    "Macron":        ["macron_trump"],
    "Merz":          ["merz_trump", "germany_trump"],
    "Starmer":       ["starmer_trump"],
    "Meloni":        ["meloni"],
    "Orbán":         [],
    "Erdoğan":       ["trump_turkey"],
    "Modi":          ["india_pakistan"],
    "Lula":          ["lula"],

    "Bolsonaro":     ["bolsonaro"],
    "MBS":           ["mbs", "israel_saudi", "saudi_arabia_usa"],
    "Yoon":          ["yoon", "south_korea_trump"],
    "Constitutional Court": ["yoon", "south_korea_trump"],
    "Ishiba":        ["south_korea_trump"],
    "Kim Jong Un":   ["north_korea"],
    "Xi Jinping":    ["china_taiwan", "xi_jinping"],
    "Jerome Powell": ["jerome_powell"],
    "Rutte":         ["nato_rutte"],
    "Sánchez":       [],
    "Guterres":      ["trump_un_general_assembly"],
    "Lai Ching-te":  ["china_taiwan"],
    "Durov":         [],
    "Pope":          ["pope"],
    "Pope Francis":  ["pope"],
    "Maduro":        ["usa_venezuela"],
    "Carney":        [],
    "Lukashenko":    [],
    "Sisi":          ["gaza_israel", "gaza_usa"],
    "Reza Pahlavi":  ["iran_israel", "iran_usa"],
    "Von der Leyen": [],
    "Trudeau":       [],
    "Obama":         [],
    "Milei":         [],
 

    #  Countries / blocs 
    "China":         ["china_taiwan", "xi_jinping"],
    "Taiwan":        ["china_taiwan"],
    "Turkey":        ["trump_turkey"],
    "South Korea":   ["south_korea_trump", "yoon"],
    "North Korea":   ["north_korea"],
    "India":         ["india_pakistan"],
    "Pakistan":      ["india_pakistan"],
    "Germany":       ["germany_trump", "merz_trump"],
    "Poland":        [],
    "UK":            ["starmer_trump"],
    "France":        ["macron_trump"],
    "Armenia":       ["armenia_trump"],
    "Venezuela":     ["usa_venezuela"],
    "Iraq":          ["iraq_israel"],
    "USA":           ["iran_usa", "gaza_usa", "saudi_arabia_usa",
                      "usa_venezuela", "trump_zelenskyy", "trump_putin"],

    "EU":            [],
    "NATO":          ["nato_rutte"],
    "NATO Public Forum Address": ["nato_rutte"],
    "Panama":        [],
    "Yemen":         ["israel_yemen", "houthi"],
    "Jordan":        ["gaza_israel"],
    "Afghanistan":   [],
    "Belarus":       [],
    "Azerbaijan":    ["armenia_trump"],
    "South Africa":  ["south_african_trump"],
 
    #  Orgs / groups 
    "UN":            ["trump_un_general_assembly", "netanyahu_unga"],
    "UN General Assembly": ["trump_un_general_assembly", "netanyahu_unga"],
    "BRICS":         [],
    "ICC":           [],
    "WHO":           [],
    "G7":            [],
    "PKK":           ["trump_turkey"],
    "SpaceX":        ["elon_musk_trump"],
    "Tesla":         ["elon_musk_trump"],
    "DeepSeek":      [],
    "TikTok":        [],
    "Nord Stream":   ["russia_ukraine"],
    "Suez Canal":    [],
    "Abraham Accord": ["israel_saudi"],
    "Warsaw Pact":   [],
 
    # ── Misc / people ─────────────────────────────────────────────────────────
    "Epstein":       [],
    "Greta Thunberg": [],
    "AI":            [],
    "Heathrow":      [],
    "Baghdad":       ["iraq_israel"],
    "South African": ["south_african_trump"],
    "Bitcoin":       [],
    "Crypto":        [],
    "Japan":         ["japan_election"],
    "House of Councillors": ["japan_election"],
    "Canada":        [],
    "Brazil":        [],
    "MrBeast":       [],
    "Baier":         [],
}




# 5. TOPIC_TO_TICKERS

# maps NER keyword → relevant stock/ETF tickers
# keys match canonical display form from KNOWN_NAMES (same as KEYWORD_TO_CORPORA)
# ordered by frequency from inspect_keywords.py output
# empty list [] = no liquid instrument to map to
#
# ticker notes:
#   LMT/RTX/NOC/GD  = US defence primes (move on conflict escalation)
#   USO             = US Oil Fund ETF (crude oil proxy)
#   XOM/CVX         = major US oil majors
#   SHEL/BP         = European energy (Russia/gas exposure)
#   WEAT/CORN       = grain futures ETFs (Ukraine = major exporter)
#   GLD             = gold (geopolitical risk hedge, spikes on escalation)
#   TLT             = 20yr US treasury ETF (rate/macro risk)
#   SPY             = S&P 500 (broad macro)
#   DXY             = US dollar index
#   TSM             = TSMC (Taiwan strait risk proxy)
#   NVDA            = Nvidia (AI + China export controls)
#   EWG/EWQ/EWU     = Germany/France/UK country ETFs
#   FEZ             = Euro Stoxx 50 ETF
#   EWZ             = Brazil ETF
#   INDA            = India ETF
#   TUR             = Turkey ETF
#   EPOL            = Poland ETF
#   005930.KS       = Samsung (South Korea political risk)
#   000660.KS       = SK Hynix (South Korea political risk)
#   CNH=X           = Chinese yuan (trade war proxy)
#   GBP=X           = British pound
#   EUR=X           = Euro
#   BTC-USD         = Bitcoin
#   ETH-USD         = Ethereum
 
TOPIC_TO_TICKERS = {
 
    # ── High frequency (100+ markets) ────────────────────────────────────────
 
    "Israel":        ["LMT", "RTX", "NOC", "GD", "GLD"],
    "Trump":         ["DXY", "GLD", "TLT", "SPY"],
    "Iran":          ["USO", "XOM", "CVX", "GLD"],
    "USA":           ["DXY", "SPY", "TLT"],
    "Russia":        ["USO", "SHEL", "BP", "WEAT", "GLD"],
    "Zelenskyy":     ["WEAT", "LMT", "RTX", "GLD"],
    "Ukraine":       ["WEAT", "CORN", "LMT", "RTX"],
    "Yemen":         ["USO", "XOM", "LMT", "RTX"],
    "Putin":         ["USO", "SHEL", "WEAT", "GLD"],
    "Gaza":          ["LMT", "RTX", "NOC", "GLD"],
 
    # ── Medium frequency (20-99 markets) ─────────────────────────────────────
 
    "Netanyahu":     ["LMT", "RTX", "GLD"],
    "Syria":         ["USO", "LMT", "GLD"],
    "Hamas":         ["LMT", "RTX", "GLD"],
    "South Korea":   ["005930.KS", "000660.KS", "LMT"],
    "India":         ["INDA", "LMT", "GLD"],
    "NATO":          ["LMT", "RTX", "NOC", "GD"],
    "UN General Assembly": ["GLD", "TLT"],
    "Pakistan":      ["LMT", "GLD"],
    "Starmer":       ["EWU", "GBP=X"],
    "Turkey":        ["TUR", "GLD"],
    "Elon Musk":     ["TSLA", "NVDA"],
    "North Korea":   ["LMT", "RTX", "005930.KS", "GLD"],
    "Saudi Arabia":  ["USO", "XOM", "CVX"],
    "China":         ["TSM", "NVDA", "USDCNH=X"],
    "Houthi":        ["USO", "LMT", "RTX"],
    "Merz":          ["EWG", "FEZ"],
    "Poland":        ["EPOL", "LMT"],
    "Xi Jinping":    ["TSM", "CNH=X", "NVDA"],
    "UK":            ["EWU", "GBP=X"],
    "Macron":        ["EWQ", "FEZ", "EUR=X"],
    "Bitcoin":       ["BTC-USD"],
    "Yoon":          ["005930.KS", "000660.KS"],
    "Lebanon":       ["LMT", "RTX", "GLD"],
    "Crypto":        ["BTC-USD", "ETH-USD"],
    "UN":            ["GLD", "TLT"],
    "EU":            ["FEZ", "EUR=X"],
    "Gulf Leaders Summit": ["USO", "XOM"],
    "Vance":         ["GLD", "TLT"],
    "Rutte":         ["LMT", "RTX", "FEZ"],
    "NATO Public Forum Address": ["LMT", "RTX"],
    "Palestine":     ["LMT", "RTX", "GLD"],
    "South African": ["GLD"],
    "Khamenei":      ["USO", "GLD"],
    "Armenia":       ["GLD"],
    "Al-Sharaa":     ["USO", "GLD"],
    "Pokrovsk":      ["WEAT", "LMT"],
    "Qatar":         ["USO", "LMT"],
    "AI":            ["NVDA", "MSFT", "GOOGL", "AMD"],
    "France":        ["EWQ", "FEZ", "EUR=X"],
    "Hezbollah":     ["LMT", "RTX", "GLD"],
    "Kim Jong Un":   ["LMT", "RTX", "GLD"],
    "Venezuela":     ["USO"],
    "Moscow":        ["WEAT", "USO", "SHEL"],
    "Germany":       ["EWG", "FEZ", "EUR=X"],
    "Lula":          ["EWZ"],
 
    # ── Lower frequency (5-19 markets) ───────────────────────────────────────
 
    "MBS":           ["USO", "XOM"],
    "Crimea":        ["WEAT", "LMT", "GLD"],
    "Brazil":        ["EWZ"],
    "Bolsonaro":     ["EWZ"],
    "Egypt":         ["GLD"],
    "Constitutional Court": ["005930.KS", "000660.KS"],
    "Siversk":       ["WEAT"],
    "Erdoğan":       ["TUR", "GLD"],
    "Iraq":          ["USO", "LMT"],
    "Maduro":        ["USO"],
    "Damascus":      ["USO", "LMT"],
    "Taiwan":        ["TSM", "NVDA"],
    "Sudzha":        ["WEAT"],
    "Kupiansk":      ["WEAT"],
    "Donbas":        ["WEAT", "LMT"],
    "Milei":         [],                    # no liquid instrument
    "Belarus":       ["GLD"],
    "Mexico":        ["MXN=X"],
    "Apple":         ["AAPL"],
    "Fordow":        ["USO", "GLD"],
    "Australia":     [],
    "Argentina":     [],
    "Von der Leyen": ["FEZ", "EUR=X"],
    "TikTok":        ["META", "GOOGL"],
    "Meloni":        ["FEZ", "EUR=X"],
    "Jordan":        ["GLD", "LMT"],
    "Greenland":     [],
    "UAE":           ["USO"],
    "Nord Stream":   ["SHEL", "BP"],
    "Afghanistan":   ["LMT", "GLD"],
    "Nasrallah":     ["LMT", "RTX", "GLD"],
    "Carney":        ["GBP=X"],             # Canada context
    "Lukashenko":    ["GLD"],
    "Hungary":       ["FEZ"],
    "Ceasefire":     ["GLD", "LMT"],
    "Spain":         ["FEZ", "EUR=X"],
    "Modi":          ["INDA", "LMT"],
 
    # ── Low frequency (2-4 markets) ───────────────────────────────────────────
 
    "Suez Canal":    ["USO", "XOM"],
    "Tesla":         ["TSLA"],
    "Reza Pahlavi":  ["USO", "GLD"],
    "Panama":        ["USO"],               # canal / shipping
    "Black Sea":     ["WEAT", "USO"],
    "Witkoff":       ["GLD"],
    "West Bank":     ["LMT", "RTX", "GLD"],
    "UNRWA":         ["GLD"],
    "DeepSeek":      ["NVDA", "AMD"],
    "Lai Ching-te":  ["TSM"],
    "White House":   ["DX-Y.NYB", "GLD"],
    "IDF":           ["LMT", "RTX"],
    "Sisi":          ["GLD"],
    "BRICS":         ["GLD", "EEM"],        # emerging markets ETF
    "Iron Dome":     ["LMT", "RTX"],
    "Tehran":        ["USO", "GLD"],
    "Hormuz":        ["USO", "XOM"],
    "DOGE":          ["TSLA"],
    "Congo":         ["CPER"],              # copper
    "Azerbaijan":    ["GLD"],
    "Marco Rubio":   ["GLD", "TLT"],
    "Sinwar":        ["LMT", "RTX"],
    "Ben Gvir":      ["LMT", "RTX"],
 
    # ── Single market / noise — leave empty ───────────────────────────────────
 
    "Epstein":       [],
    "Greta Thunberg": [],
    "MrBeast":       [],
    "Pope":          [],
    "Pope Francis":  [],
    "Baier":         [],
    "Heathrow":      [],
    "Guterres":      [],
    "Durov":         [],
    "Sánchez":       [],
    "Ishiba":        [],
    "Trudeau":       [],
    "Obama":         [],
    "Biden":         [],
    "Japan":         [],                    # house_of_councillors markets, no clear proxy
    "House of Councillors": [],
    "Canada":        [],
    "Thailand":      [],
    "Cambodia":      [],
    "Indonesia":     [],
    "Philippines":   [],
    "Finland":       [],
    "Norway":        [],
    "Ireland":       [],
    "Netherlands":   [],
    "Belgium":       [],
    "Austria":       [],
    "Portugal":      [],
    "Serbia":        [],
    "Greece":        [],
    "Vatican":       [],
    "Gold":          ["GLD"],
    "Oil":           ["USO", "XOM", "CVX"],
    "Nvidia":        ["NVDA"],
    "SpaceX":        ["TSLA"],
    "Semiconductor": ["NVDA", "TSM", "AMD"],
    "LNG":           ["SHEL", "BP"],
    "Kharg":         ["USO"],
    "Persian Gulf":  ["USO", "XOM"],
    "Baghdad":       ["USO", "LMT"],
    "Kyiv":          ["WEAT", "LMT"],
    "Donetsk":       ["WEAT"],
    "PKK":           ["TUR"],
    "Zangezur Corridor": ["GLD"],
    "Abraham Accord": ["LMT", "USO"],
    "Warsaw Pact":   ["LMT", "RTX"],
    "G7":            ["DXY", "GLD"],
    "WTO":           ["DXY"],
    "WHO":           [],
    "ICC":           [],
    "xAI":           ["TSLA"],
    "Grok":          ["TSLA"],
    "Starlink":      ["TSLA"],
}