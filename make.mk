PY = python3
PY_SCRIPT = run_and_plot.py
BIN = central_comando

.PHONY: run clear

# run -> inicia o script Python
run:
	$(PY) $(PY_SCRIPT)

# clear -> remove executáveis/binários
clear:
	-rm -f $(BIN) $(BIN).exe
	@echo "Executáveis removidos."