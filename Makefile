# usage: make <target>
# "make pipeline" - for full pipeline

.PHONY: download markets tokens filter process db queries pipeline clean

# download raw snapshot 
download:
	curl -L -o orderFilled_complete.csv.xz \
		https://polydata-archive.s3.us-east-1.amazonaws.com/orderFilled_complete.csv.xz

# each step of pipeline 
markets:
	python src/pipeline/collect_markets.py

tokens:
	python src/pipeline/fetch_tokens.py

filter:
	python src/pipeline/filter_xz.py

process:
	python src/pipeline/process_trades.py

db:
	python src/pipeline/load_db.py

queries:
	python src/pipeline/query_tests.py

# full pipeline 
pipeline: download markets tokens filter process db queries

# utility
clean:
	rm -f data/processed/trades_clean.csv
	rm -f data/analytical/polymarket.ddb