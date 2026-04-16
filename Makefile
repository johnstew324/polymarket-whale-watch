# usage: make <target>
# "make pipeline" - for full pipeline

.PHONY: download markets tokens filter process db queries pipeline clean \
        clean-sentiment clean-proquest clean-scored clean-all clean-notebooks \
        setup report open-report clean-report rebuild-report figures loc


# 1. data collection and processing
# download raw snapshot 
download:
	curl -L -o data/raw/orderFilled_complete.csv.xz \
		https://polydata-archive.s3.us-east-1.amazonaws.com/orderFilled_complete.csv.xz

# each step of pipeline 
markets:
	python -m src.pipeline.collect_markets

tokens:
	python -m src.pipeline.fetch_tokens

filter:
	python -m src.pipeline.filter_xz

process:
	python -m src.pipeline.process_trades

db:
	python -m src.pipeline.load_db

queries:
	python -m src.pipeline.query_tests

# full pipeline 
pipeline: download markets tokens filter process db queries




# 2. sentiment pipeline

# parse raw ProQuest txt files into per-topic CSVs
parse-proquest:
	python -m src.sentiment.parse_proquest
 
# run FinBERT scoring overnight - skips already-scored topics
score-ft:
	python -m src.sentiment.finbert_scorer
 
# aggregate pre-scored articles into weekly sentiment per market
collect-ft:
	python -m src.sentiment.collect_ft
 
# run Truth Social sentiment pipeline
collect-ts:
	python -m src.sentiment.collect_truth_social

# pull yfinance weekly financial signals
collect-financial:
	python -m src.sentiment.collect_financial
 
# load sentiment + financial CSVs into DuckDB
load-sentiment:
	python -m src.sentiment.load_sentiment_financial_db

# coverage report - how many markets have sentiment/financial data
inspect-coverage:
	python -m src.sentiment.inspect_coverage


 
# full FT pipeline (parse -> score -> collect -> load)
sentiment-ft: parse-proquest score-ft collect-ft load-sentiment
 
# full Truth Social pipeline
sentiment-ts: collect-ts load-sentiment

# full financial pipeline
sentiment-financial: collect-financial load-sentiment

# todo : maybe pytrends

# full sentiment pipeline (both sources)
sentiment: sentiment-ts sentiment-ft sentiment-financial


# include analysis pipeline
# so queries and wallet analysis
# then wordclouds LDA etc.. 


# 3. cleanup
 
# remove all processed trade data (re-run pipeline to rebuild)
clean:
	rm -f data/processed/trades_clean.csv
	rm -f data/analytical/polymarket.ddb
 
# remove sentiment output CSVs (re-run collect scripts to rebuild)
clean-sentiment:
	rm -f data/processed/sentiment/truth_social_sentiment.csv
	rm -f data/processed/sentiment/ft_sentiment.csv
 
# remove parsed ProQuest CSVs (re-run parse-proquest to rebuild)
# use this when adding new .txt files for a topic
clean-proquest:
	rm -f data/processed/proquest/*_articles.csv
 

# remove scored ProQuest CSVs (re-run score-ft to rebuild)
# use this when re-downloading articles for a topic
clean-scored:
	rm -f data/processed/proquest/*_articles_scored.csv
 
# remove everything: full rebuild from raw data
clean-all: clean clean-sentiment clean-proquest clean-scored
 

clean-notebooks:
	jupyter nbconvert --clear-output --inplace notebooks/*.ipynb
 

# 4. setup

setup:
	pip install -r requirements.txt
	python -m spacy download en_core_web_sm




# 5. report
report:
	cd report && latexmk -pdf -interaction=nonstopmode main.tex

clean-report:
	cd report && latexmk -c
