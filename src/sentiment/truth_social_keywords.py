# sentiment/truth_social_keywords.py
#
# Maps Polymarket market tags to search keywords for Truth Social post matching.
# Proper nouns and specific terms only — no generic words like "war" or "attack"
# which would match almost every post and dilute the signal.

TAG_KEYWORDS = {
    # catch-alls — kept minimal, only very specific terms
    "geopolitics":               [],  # too broad, skip
    "politics":                  [],  # too broad, skip
    "world":                     [],  # too broad, skip
    "all":                       [],  # skip
    "breaking-news":             [],  # skip
    "world-affairs":             [],  # skip
    "world-news":                [],  # skip
    "global-politics":           [],  # skip
    "foreign-policy":            [],  # skip
    "global-elections":          [],  # skip
    "elections":                 [],  # skip
    "world-elections":           [],  # skip
    "mention-markets":           [],  # skip
    "military-action":           [],  # skip

    # trump
    "trump":                     ["Trump", "MAGA", "Mar-a-Lago"],
    "trump-presidency":          ["Trump", "executive order", "White House"],

    # russia / ukraine
    "russia":                    ["Russia", "Putin", "Kremlin", "Moscow"],
    "ukraine":                   ["Ukraine", "Zelensky", "Kyiv", "Zelenskyy"],
    "putin":                     ["Putin", "Kremlin"],
    "ukraine-map":               ["Ukraine", "Donbas", "Zaporizhzhia", "Kherson", "Kyiv"],
    "ukraine-peace-deal":        ["Ukraine", "Zelensky", "Zelenskyy", "peace deal", "ceasefire"],

    # iran
    "iran":                      ["Iran", "Khamenei", "Tehran", "IRGC"],
    "us-iran":                   ["Iran", "Tehran", "IRGC"],
    "trump-iran":                ["Iran", "Tehran"],
    "khamenei":                  ["Khamenei", "supreme leader", "Iran"],
    "iranian-leadership-regime": ["Khamenei", "Iran", "supreme leader", "Tehran"],
    "israel-x-iran":             ["Israel", "Iran"],
    "nuclear":                   ["nuclear", "uranium", "warhead", "enrichment"],
    "nuclear-weapons":           ["nuclear", "warhead", "nuke"],
    "nuclear-deal":              ["nuclear deal", "Iran", "uranium"],

    # israel / gaza / hezbollah / hamas
    "israel":                    ["Israel", "Netanyahu", "IDF"],
    "gaza":                      ["Gaza", "Hamas", "Palestinian"],
    "hezbollah":                 ["Hezbollah", "Nasrallah"],
    "hamas":                     ["Hamas", "Gaza", "hostage"],
    "daily-strikes":             ["Israel", "IDF", "airstrike"],
    "lebanon":                   ["Lebanon", "Beirut", "Hezbollah"],

    # china / taiwan
    "china":                     ["China", "Xi Jinping", "Beijing", "CCP"],
    "taiwan":                    ["Taiwan", "Taipei"],

    # houthis / shipping
    "houthis":                   ["Houthi", "Yemen", "Sanaa"],
    "houthi":                    ["Houthi", "Yemen"],
    "strait-of-hormuz":          ["Hormuz", "Iran", "tanker"],
    "suez-canal":                ["Suez", "Houthi", "Red Sea"],
    "oil":                       ["OPEC", "crude oil", "oil prices"],

    # yemen
    "yemen":                     ["Yemen", "Houthi", "Sanaa"],

    # india / pakistan
    "india":                     ["India", "Modi", "New Delhi"],
    "india-pakistan":            ["India", "Pakistan", "Kashmir"],
    "pakistan":                  ["Pakistan", "Islamabad"],

    # north korea
    "north-korea":               ["North Korea", "Kim Jong", "Pyongyang"],

    # venezuela
    "venezuela":                 ["Venezuela", "Maduro", "Caracas"],
    "cartel":                    ["cartel", "fentanyl", "Mexico"],
    "mexico":                    ["Mexico", "cartel", "border"],

    # middle east (broad)
    "middle-east":               ["Middle East", "Gaza", "Iran", "Israel"],
}