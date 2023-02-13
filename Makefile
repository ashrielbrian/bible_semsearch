VENV = .venv
BIN = ${VENV}/bin
PYTHON = ${BIN}/python

init:
	python -m venv ${VENV}
	${PYTHON} -m pip install -U -r requirements.txt

embeddings:
	chmod +x ./encode.sh && ./encode.sh

clean:
	rm -rf ${VENV}