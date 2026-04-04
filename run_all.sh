#!/bin/bash
# Executa todos os experimentos em sequência.
# Os resultados são salvos em experiments/results/
# Tempo estimado: ~6 horas em GPU NVIDIA RTX 5090

echo "=== Experimento 1: 40% atacantes, 100% inversão ==="
python experiments/exp1.py

echo "=== Experimento 2: 60% atacantes, 100% inversão ==="
python experiments/exp2.py

echo "=== Experimento 3: 40% atacantes, 60% inversão ==="
python experiments/exp3.py

echo "=== Experimento 4: 40% atacantes, 40% inversão ==="
python experiments/exp4.py

echo "=== Experimento 5: 40% atacantes, 100% inversão, K=10 ==="
python experiments/exp5.py

echo "=== Experimento 6: 40% atacantes, 100% inversão, K=35 ==="
python experiments/exp6.py

echo "=== Todos os experimentos concluídos ==="
