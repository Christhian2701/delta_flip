import json
import matplotlib.pyplot as plt

# 1. Carregar os dados do arquivo JSON
arquivo_json = 'new_results/metrics_with_delta.json'

with open(arquivo_json, 'r') as f:
    dados = json.load(f)

# 2. Extrair as listas de métricas
rounds = []
test_accuracy = []
delta_accuracy = []
test_loss = []
delta_loss = []

# Iterar sobre a lista de resultados da chave "fedavg"
for rodada in dados['flips']:
    rounds.append(rodada['round'])
    test_accuracy.append(rodada['test_accuracy'])
    delta_accuracy.append(rodada['delta_accuracy'])
    test_loss.append(rodada['test_loss'])
    delta_loss.append(rodada['delta_loss'])

    print(f"Round {rodada['round']}: Test Acc = {rodada['test_accuracy']:.4f}, Delta Acc = {rodada['delta_accuracy']:.4f}, Test Loss = {rodada['test_loss']:.4f}, Delta Loss = {rodada['delta_loss']:.4f}")

# 3. Configurar a figura e os subgráficos (1 linha, 2 colunas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# --- Gráfico 1: Acurácia ---
ax1.plot(rounds, test_accuracy, marker='o', linestyle='-', color='blue', label='Modelo Global (Test Acc)')
ax1.plot(rounds, delta_accuracy, marker='s', linestyle='--', color='orange', label='Modelo Delta (Delta Acc)')

ax1.set_title('Comparação de Acurácia: Normal vs Delta')
ax1.set_xlabel('Rodadas (Rounds)')
ax1.set_ylabel('Acurácia')
ax1.set_xticks(rounds) # Garante que o eixo X mostre números inteiros das rodadas
ax1.grid(True, linestyle=':', alpha=0.7)
ax1.legend()

# --- Gráfico 2: Loss (Perda) ---
ax2.plot(rounds, test_loss, marker='o', linestyle='-', color='red', label='Modelo Global (Test Loss)')
ax2.plot(rounds, delta_loss, marker='s', linestyle='--', color='green', label='Modelo Delta (Delta Loss)')

ax2.set_title('Comparação de Loss: Normal vs Delta')
ax2.set_xlabel('Rodadas (Rounds)')
ax2.set_ylabel('Loss (Erro)')
ax2.set_xticks(rounds)
ax2.grid(True, linestyle=':', alpha=0.7)
ax2.legend()

# 4. Ajustar o layout para não sobrepor textos e mostrar o gráfico
plt.tight_layout()

# Se quiser salvar a imagem em um arquivo, descomente a linha abaixo:
# plt.savefig('comparacao_metricas.png', dpi=300)

caminho_salvamento = 'new_results/grafico_comparativo.png'
plt.savefig(caminho_salvamento, dpi=300, bbox_inches='tight')
print(f"Gráfico salvo com sucesso em: {caminho_salvamento}")
