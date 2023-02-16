VENV = .venv
BIN = ${VENV}/bin
PYTHON = ${BIN}/python

init:
	python -m venv ${VENV}
	${PYTHON} -m pip install -U -r requirements.txt

embeddings:
	chmod +x ./scripts/encode.sh && ./scripts/encode.sh

index:
	chmod +x ./scripts/pinecone.sh && ./scripts/pinecone.sh

clean:
	rm -rf ${VENV}